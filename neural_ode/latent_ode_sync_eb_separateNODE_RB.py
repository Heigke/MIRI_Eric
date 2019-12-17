import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--adjoint", type=eval, default=True)
parser.add_argument("--visualize", type=eval, default=True)
parser.add_argument("--niters", type=int, default=2000)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--noise", type=int, default=0.15)
parser.add_argument("--end_time", type=int, default=40)
parser.add_argument("--samples", type=int, default=100)
parser.add_argument("--async_data", type=int, default=True)
parser.add_argument("--vis_freq", type=int, default=10)
args = parser.parse_args()

# Another way of calculating the gradients
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def curve_function(timestamps, func_type):
    if func_type == 0:
        return np.sin(timestamps) * np.exp(-0.1 * timestamps)
    elif func_type == 1:
        return np.cos(0.5 * timestamps) * np.exp(-0.05 * timestamps)


def generate_curve(
    func_type,
    ncurves=1000,
    ntotal=500,
    nsample=100,
    start=0.0,
    stop=40,
    noise_std=0.1,
    savefig=True,
    async_data=True,
):
    """ Generate one curve
    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check.

      Returns:
        Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
        second element is noisy observations of size (nspiral, nsample, 2),
        third element is timestamps of size (ntotal,),
        and fourth element is timestamps of size(nsample,) """
    if async_data:
        # Create async. time tensor
        time_rnd = np.random.random_sample(nsample)
        time_rnd = np.sort(time_rnd)
        time_rnd = time_rnd * stop
        time = time_rnd
    else:
        # Or sync. time tensor
        time = np.linspace(0, stop, nsample)

    # Make sure that starting time 0 is the first time
    time[0] = 0.0
    # Create cont. time tensor
    time_cont = np.linspace(0, stop, ntotal)

    # Generate continuous y values
    true_y_cont = curve_function(time_cont, func_type)

    # Generate noisy y values
    dt = time[1:nsample] - time[0 : nsample - 1]
    dt = np.insert(dt, 0, 0)
    true_y_dt = []
    for _ in range(ncurves):
        noise = args.noise * (np.random.rand(nsample) - 0.5) * 2
        new_true_y = curve_function(time, func_type) + noise
        # One of the inputs is the diff. in time between points.
        new_true_dt = dt
        new_true_y_dt = np.stack((new_true_y, new_true_dt), axis=1)
        true_y_dt.append(new_true_y_dt)

    true_y_dt = np.asarray(true_y_dt)
    return true_y_cont, true_y_dt, time_cont, time


def create_plot():
    # Create plot
    if args.visualize:
        figSamp = plt.figure(figsize=(12, 4), facecolor="white")
        ax_traj_samp = figSamp.add_subplot(132, frameon=False)
        ax_traj_samp.cla()  # Clears
        ax_traj_samp.set_title("Ground Truth N' Samples")
        ax_traj_samp.set_xlabel("Time")
        ax_traj_samp.set_ylabel("Curves")
        return figSamp, ax_traj_samp


def plot_data(fig, subplt, y_values, timestamps, line_type, label_name):
    plt.plot(timestamps, y_values, line_type, label=label_name)
    fig.tight_layout()
    plt.legend()
    plt.draw()


# Plot some of the sensors timestamps vs each other
height_async_data = 0


def plot_unsync_sensors(timestamps, label_name):
    global height_async_data
    plt.plot(
        timestamps, height_async_data * np.ones(timestamps.size), "o", label=label_name
    )
    plt.ylim(-0.05, 2 * height_async_data)
    plt.legend()
    plt.draw()
    height_async_data += 0.05


def save_plot(file_name):
    plt.savefig("latent_ode_sync_fig/" + file_name + ".png", dpi=500)


# NODE in the latent space
class LatentNode(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentNode, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


# Bringing input in Y domain to lantent space Z with RNN
class RnnEncoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RnnEncoder, self).__init__()
        self.nhidden = nhidden
        self.batch = nbatch
        self.input_to_hidden = nn.Linear(obs_dim + nhidden, nhidden)
        self.hidden_to_output = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        hidden_state = torch.tanh(self.input_to_hidden(combined))
        out = self.hidden_to_output(hidden_state)
        return out, hidden_state

    def init_hidden_state(self):
        return torch.zeros(self.batch, self.nhidden)


# Since NODE are space preserving we need to decode.
# Brings new points in latent Z space to Y space.
class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=3, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Calculating average loss for better overview of training
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


# Creating a pick from logarithmic normal probability distribution
# Basically providing the log of a relative likelihood that the value of the
# random variable would equal that sample.(wikipedia: Probability density function)
def log_normal_prob_dist(x, mean, logvar):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar))


# Kullback-Leibler divergence between a diagonal multivariate normal,
# and a normal distribution with specified mu and logvar.
# (Wikipedia: Kullback-Leibler Divergence, examples: Multivariate normal distribution)
def kullback_leibler_divergence_normaldist(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.0
    lstd2 = lv2 / 2.0

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2)) - 0.5
    return kl


# Combine timestamps into a single tensor
def combine_timestamps(t1, t2):
    t1 = t1.cpu().numpy()
    t2 = t2.cpu().numpy()
    indx1 = []
    indx2 = []
    combine_timestamps = np.concatenate((t1, t2), axis=0)
    combine_timestamps.sort()
    # Remove one of the common timepoints 0.0
    combine_timestamps = np.delete(combine_timestamps, 0)
    # Find index for the both original time tensors t1 and t2
    boolean_match1 = np.in1d(combine_timestamps, t1)
    boolean_match2 = np.in1d(combine_timestamps, t2)
    indx = np.linspace(0, combine_timestamps.size - 1, combine_timestamps.size)
    indx_t1 = indx[boolean_match1]
    indx_t2 = indx[boolean_match2]
    indx_t1 = indx_t1.astype(int)
    indx_t2 = indx_t2.astype(int)
    return combine_timestamps, indx_t1, indx_t2


# Fills in the holes in the original data
def forward_rnn_data(pred_x_A, pred_x_B, time_indx_A, time_indx_B, true_y_A, true_y_B):
    pred_x_A[:, time_indx_A] = true_y_A
    pred_x_B[:, time_indx_B] = true_y_B
    return pred_x_A, pred_x_B


if __name__ == "__main__":
    latent_dim_total = 4
    latent_dim_sensors = 2
    nhidden = 20
    rnn_nhidden = 25
    obs_dim_total = 2  # A and B sensors
    obs_dim_sensors = 2  # Sensor value and dt
    ncurves = 1000
    noise_std = 0.3
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    # Create folder if vis is true
    current_time = datetime.datetime.now()
    str_current_day = (
        str(current_time.year)
        + "_"
        + str(current_time.month)
        + "_"
        + str(current_time.day)
        + "_"
        + str(current_time.hour)
        + "_"
        + str(current_time.minute)
    )
    directory_name = "async_unsync_" + str_current_day
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    # Generate two curves with different time vectors
    # Plot ground truth
    true_y_cont_A, true_y_dt_A, time_cont_A, time_A = generate_curve(func_type=0, ncurves=ncurves)
    fig, subplt = create_plot()
    plot_data(fig, subplt, true_y_cont_A, time_cont_A, "-", "Cont. A")
    true_y_cont_B, true_y_dt_B, time_cont_B, time_B = generate_curve(func_type=1, ncurves=ncurves)
    plot_data(fig, subplt, true_y_cont_B, time_cont_B, "-", "Cont. B")
    plot_data(fig, subplt, true_y_dt_A[0, :, 0], time_A, "o", "Samples A")
    plot_data(fig, subplt, true_y_dt_B[0, :, 0], time_B, "o", "Samples B")
    save_plot("ground_truth_samples")

    # Plot a sample to see the unsync. curves
    sample_A = time_A[0:10]
    sample_B = time_B[0:10]
    fig_time, subplt_time = create_plot()
    plot_unsync_sensors(sample_A, "Sensor A")
    plot_unsync_sensors(sample_B, "Sensor B")
    save_plot("unsync_sensors")

    # Make data into torch tensors at GPU, if possible
    true_y_cont_A = torch.from_numpy(true_y_cont_A).float().to(device)
    true_y_dt_A = torch.from_numpy(true_y_dt_A).float().to(device)
    time_cont_A = torch.from_numpy(time_cont_A).float().to(device)
    time_A = torch.from_numpy(time_A).float().to(device)
    true_y_A = true_y_dt_A[:, :, 0]

    true_y_cont_B = torch.from_numpy(true_y_cont_B).float().to(device)
    true_y_dt_B = torch.from_numpy(true_y_dt_B).float().to(device)
    time_cont_B = torch.from_numpy(time_cont_B).float().to(device)
    time_B = torch.from_numpy(time_B).float().to(device)
    true_y_B = true_y_dt_B[:, :, 0]

    # Fetching the common timestamps from sensors for prediction
    # Calculates the dt for forward rnn
    common_timestamps, time_indx_A, time_indx_B = combine_timestamps(time_A, time_B)
    common_timestamps = torch.from_numpy(common_timestamps).float().to(device)
    size = common_timestamps.size(0)
    common_dt = common_timestamps[1:size] - common_timestamps[0 : size - 1]
    common_dt = common_dt.cpu()
    common_dt = np.insert(common_dt, 0, 0)
    common_dt = torch.ones((ncurves, args.samples * 2 - 1)) * common_dt
    common_dt = common_dt.to(device)

    # Create models
    latent_node = LatentNode(latent_dim_total, nhidden).to(device)
    latent_node_fwd = LatentNode(latent_dim_total, nhidden).to(device)
    rnn_encoder_A = RnnEncoder(
        latent_dim_sensors, obs_dim_sensors, rnn_nhidden, ncurves
    ).to(device)
    rnn_encoder_B = RnnEncoder(
        latent_dim_sensors, obs_dim_sensors, rnn_nhidden, ncurves
    ).to(device)
    decoder = Decoder(latent_dim_total, obs_dim_total, nhidden).to(device)
    decoder_fwd = Decoder(latent_dim_total, obs_dim_total, nhidden).to(device)
    rnn_encoder_forw = RnnEncoder(
        latent_dim_total, obs_dim_total + 1, rnn_nhidden, ncurves
    ).to(device)
    params = (
        list(latent_node.parameters())
        + list(rnn_encoder_A.parameters())
        + list(rnn_encoder_B.parameters())
        + list(decoder.parameters())
    )
    optimizer = optim.Adam(params, lr=args.lr)
    # optimizer = optim.RMSprop(params, lr=1e-3)
    loss_meter = RunningAverageMeter()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        # Run backwards in time through encoder for both A and B sensor
        h_A = rnn_encoder_A.init_hidden_state().to(device)
        h_B = rnn_encoder_B.init_hidden_state().to(device)

        training_times_A = reversed(range(int(np.trunc(true_y_dt_A.size(1)))))
        training_times_B = reversed(range(int(np.trunc(true_y_dt_B.size(1)))))
        # Run through sensor A
        for t in training_times_A:
            obs_A = true_y_dt_A[:, t, :]
            out_A, h_A = rnn_encoder_A.forward(obs_A, h_A)
        # picking out mean and log variance for A sensor
        qz0_mean_A, qz0_logvar_A = (
            out_A[:, :latent_dim_sensors],
            out_A[:, latent_dim_sensors:],
        )

        # Run through sensor B
        for t in training_times_B:
            obs_B = true_y_dt_B[:, t, :]
            out_B, h_B = rnn_encoder_B.forward(obs_B, h_B)
        # picking out mean and log variance for B sensor
        qz0_mean_B, qz0_logvar_B = (
            out_B[:, :latent_dim_sensors],
            out_B[:, latent_dim_sensors:],
        )

        qz0_mean_comb = torch.cat((qz0_mean_A, qz0_mean_B), dim=1)
        qz0_logvar_comb = torch.cat((qz0_logvar_A, qz0_logvar_B), dim=1)
        epsilon = torch.randn(qz0_mean_comb.size()).to(device)
        z0 = epsilon * torch.exp(0.5 * qz0_logvar_comb) + qz0_mean_comb

        # Sending z0 to odeint to integrate 'sample' timesteps forward
        # Permute just permutes the tensor.

        pred_z = odeint(latent_node, z0, common_timestamps).permute(1, 0, 2)
        pred_x = decoder(pred_z)

        # We can only correct our predictions against known values
        # so we pick out the known values from pred_x_A and _B
        pred_x_A = pred_x[:, :, 0]
        pred_x_B = pred_x[:, :, 1]
        pred_x_A_known = pred_x_A[:, time_indx_A]
        pred_x_B_known = pred_x_B[:, time_indx_B]

        # Compute loss
        # First create noise as log var then call log_normal_pdf
        # samp_trajs acts as our x, pred_x as our mean i.o.w mu, noise_logvar as our logvar
        # Create logpx for A and B sensors
        noise_std_A = torch.zeros(pred_x_A_known.size()).to(device) + noise_std
        noise_std_B = torch.zeros(pred_x_B_known.size()).to(device) + noise_std
        noise_logvar_A = 2.0 * torch.log(noise_std_A).to(device)
        noise_logvar_B = 2.0 * torch.log(noise_std_B).to(device)

        logpx_A = log_normal_prob_dist(true_y_A, pred_x_A_known, noise_logvar_A).sum(-1)
        logpx_B = log_normal_prob_dist(true_y_B, pred_x_B_known, noise_logvar_B).sum(-1)

        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        analytic_kl = kullback_leibler_divergence_normaldist(
            qz0_mean_comb, qz0_logvar_comb, pz0_mean, pz0_logvar
        ).sum(-1)

        # Take the original data and fill in the gaps with predictions
        pred_true_A, pred_true_B = forward_rnn_data(
            pred_x_A, pred_x_B, time_indx_A, time_indx_B, true_y_A, true_y_B
        )
        # We also need to create a dt vector
        pred_true_AB = torch.stack((pred_true_A, pred_true_B, common_dt), dim=2)

        # #We also need to create a guidance of the rnn_encoder_forw and decoder to generalize better
        # h_forward = rnn_encoder_forw.init_hidden_state().to(device)
        # pred_true_AB_samples = pred_true_AB[:, 0 : args.samples, :]
        # max_time = int(np.trunc(pred_true_AB_samples.size(1)))
        # random_stop_time = np.random.randint(low=0, high=max_time, size=1)[0]
        # training_times_forward = range(random_stop_time)

        # for t in training_times_forward:
        #     obs_forward = pred_true_AB_samples[:, t, :]
        #     out_forward, h_forward = rnn_encoder_forw.forward(obs_forward, h_forward)
        # # picking out mean and log variance
        # qz0_mean_forward, qz0_logvar_forward = (
        #     out_forward[:, :latent_dim_total],
        #     out_forward[:, latent_dim_total:],
        # )

        # epsilon = torch.randn(qz0_mean_forward.size()).to(device)
        # z0 = epsilon * torch.exp(0.5 * qz0_logvar_forward) + qz0_mean_forward

        # y0 = decoder(z0)
        # y0_correct = pred_true_AB[:, random_stop_time-1, 0:2]
        # loss_enc_dec = (y0 - y0_correct).sum(-1)
        
        
        loss = torch.mean(
            (-logpx_A) + (-logpx_B) + analytic_kl,
            dim=0,
        )
        #+ loss_enc_dec   - Can be added to loss for training encoder decoder
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        print("Iter: {}, running avg elbo: {:.4f}".format(itr, -loss_meter.avg))

        #
        if args.visualize and itr % args.vis_freq == 0:
            print("vizing")
            with torch.no_grad():

                h_A = rnn_encoder_A.init_hidden_state().to(device)
                h_B = rnn_encoder_B.init_hidden_state().to(device)

                training_times_A = reversed(range(int(np.trunc(true_y_dt_A.size(1)))))
                training_times_B = reversed(range(int(np.trunc(true_y_dt_B.size(1)))))
                # Run through sensor A
                for t in training_times_A:
                    obs_A = true_y_dt_A[:, t, :]
                    out_A, h_A = rnn_encoder_A.forward(obs_A, h_A)
                # picking out mean and log variance for A sensor
                qz0_mean_A, qz0_logvar_A = (
                    out_A[:, :latent_dim_sensors],
                    out_A[:, latent_dim_sensors:],
                )

                # Run through sensor B
                for t in training_times_B:
                    obs_B = true_y_dt_B[:, t, :]
                    out_B, h_B = rnn_encoder_B.forward(obs_B, h_B)
                # picking out mean and log variance for B sensor
                qz0_mean_B, qz0_logvar_B = (
                    out_B[:, :latent_dim_sensors],
                    out_B[:, latent_dim_sensors:],
                )

                qz0_mean_comb = torch.cat((qz0_mean_A, qz0_mean_B), dim=1)
                qz0_logvar_comb = torch.cat((qz0_logvar_A, qz0_logvar_B), dim=1)
                epsilon = torch.randn(qz0_mean_comb.size()).to(device)
                z0 = qz0_mean_comb

                # Saving prediction to file
                pred_z_all = odeint(latent_node, z0, time_cont_A)
                pred_x_all = decoder(pred_z_all)
                print("saving files")
                torch.save(pred_x_all, directory_name + "/pred_x_all"+str(itr)+".pt")
                torch.save(time_cont_A, directory_name + "/time_all"+str(itr)+".pt")
                print("files saved")

                # Sending z0 to odeint to integrate 'sample' timesteps forward
                # Permute just permutes the tensor.
                # First traject for vis.
                z0 = z0[0]
                pred_z = odeint(latent_node, z0, time_cont_A)
                pred_x = decoder(pred_z)
                
                # Predicting into the future with rnn forward!
                h_forward = rnn_encoder_forw.init_hidden_state().to(device)
                pred_times_forward = range(args.samples-1)
                pred_true_AB_dt_correct = pred_true_AB[:, args.samples : args.samples * 2 - 1, :]
                for t in pred_times_forward:
                    obs_forward = pred_true_AB_dt_correct[:, t, :]
                    out_forward, h_forward = rnn_encoder_forw.forward(obs_forward, h_forward)
                # picking out mean and log variance
                qz0_mean_forward, qz0_logvar_forward = (
                    out_forward[:, :latent_dim_total],
                    out_forward[:, latent_dim_total:],
                )

                epsilon = torch.randn(qz0_mean_forward.size()).to(device)
                #z0 = epsilon * torch.exp(0.5 * qz0_logvar_forward) + qz0_mean_forward
                z0 = qz0_mean_forward
                z0 = z0[0]
                # Creating time vector of timepoints we want to evaluate in the untrained half
                # Picks out the times from the first timepoint after we stopped the RNN forward
                # and then subtracts the time at which we stopped.
                t_forward_pred = torch.linspace(0,args.end_time,steps=100)

                pred_z = odeint(latent_node, z0, t_forward_pred)
                pred_x_forward = decoder(pred_z)



                plt.figure()
                plt.plot(
                    time_cont_A.cpu().detach().numpy(),
                    pred_x.cpu().detach().numpy(),
                    label="Predictions Back-RNN",
                )
                plt.plot(
                    time_cont_A.cpu().detach().numpy(),
                    true_y_cont_A.cpu().detach().numpy(),
                    label="Ground truth A",
                )
                plt.plot(
                    time_cont_B.cpu().detach().numpy(),
                    true_y_cont_B.cpu().detach().numpy(),
                    label="Ground truth B",
                )
                #plt.plot((t_forward_pred+args.end_time).cpu().detach().numpy(), pred_x_forward.cpu().detach().numpy(), '-', label="RNN forward prediction")
                plt.axvspan(0, 20, facecolor="b", alpha=0.5)
                plt.legend()
                plt.draw()
                plt.savefig(
                    directory_name+"/ground_truth_with_pred" + str(itr) + ".png",
                    dpi=500,
                )
                plt.close()



    params = (
        list(latent_node_fwd.parameters())
        + list(rnn_encoder_forw.parameters())
        + list(decoder_fwd.parameters())
    )
    optimizer = optim.Adam(params, lr=args.lr)
    # optimizer = optim.RMSprop(params, lr=1e-3)
    loss_meter = RunningAverageMeter()
    pred_true_AB = pred_true_AB.detach()
    common_timestamps = common_timestamps.detach()
    print("****Beginning forward training****")
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        # Only take half of the samples to run forward rnn over so
        # we have some space for prediction.
        pred_true_AB_samples = pred_true_AB[:, 0 : args.samples, :]
        # *2-1 cause we have a common point in both sensors, t=0
        rnd_stop = int(np.random.rand(1)*(args.samples-10))+10
        pred_true_AB_correct = pred_true_AB[:, rnd_stop : args.samples * 2 - 1, 0:2]
        h_forward = rnn_encoder_forw.init_hidden_state().to(device)
        training_times_forward = range(rnd_stop)

        for t in training_times_forward:
            obs_forward = pred_true_AB_samples[:, t, :]
            out_forward, h_forward = rnn_encoder_forw.forward(obs_forward, h_forward)
        # picking out mean and log variance
        qz0_mean_forward, qz0_logvar_forward = (
            out_forward[:, :latent_dim_total],
            out_forward[:, latent_dim_total:],
        )

        epsilon = torch.randn(qz0_mean_forward.size()).to(device)
        z0 = epsilon * torch.exp(0.5 * qz0_logvar_forward) + qz0_mean_forward
        # Creating time vector of timepoints we want to evaluate in the untrained half
        # Picks out the times from the first timepoint after we stopped the RNN forward
        # and then subtracts the time at which we stopped.
        t_forward_odeint = common_timestamps[rnd_stop : common_timestamps.size(0)]
        t_forward_odeint = t_forward_odeint - common_timestamps[rnd_stop - 1]

        pred_z = odeint(latent_node_fwd, z0, t_forward_odeint).permute(1, 0, 2)
        pred_x = decoder_fwd(pred_z)

        noise_std_forward = torch.zeros(pred_x.size()).to(device) + noise_std
        noise_logvar_forward = 2.0 * torch.log(noise_std_forward).to(device)

        logpx_forward = log_normal_prob_dist(
            pred_true_AB_correct, pred_x, noise_logvar_forward
        ).sum(-1).sum(-1)

        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        analytic_kl_forward = kullback_leibler_divergence_normaldist(
            qz0_mean_forward, qz0_logvar_forward, pz0_mean, pz0_logvar
        ).sum(-1)
        
        loss = torch.mean(
            (-logpx_forward) + analytic_kl_forward,
            dim=0,
        )
        #+ loss_enc_dec   - Can be added to loss for training encoder decoder
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        print("Iter: {}, running avg elbo: {:.4f}".format(itr, -loss_meter.avg))
    
        if args.visualize and itr % args.vis_freq == 0:
            print("vizing")
            with torch.no_grad():

                h_A = rnn_encoder_A.init_hidden_state().to(device)
                h_B = rnn_encoder_B.init_hidden_state().to(device)

                training_times_A = reversed(range(int(np.trunc(true_y_dt_A.size(1)))))
                training_times_B = reversed(range(int(np.trunc(true_y_dt_B.size(1)))))
                # Run through sensor A
                for t in training_times_A:
                    obs_A = true_y_dt_A[:, t, :]
                    out_A, h_A = rnn_encoder_A.forward(obs_A, h_A)
                # picking out mean and log variance for A sensor
                qz0_mean_A, qz0_logvar_A = (
                    out_A[:, :latent_dim_sensors],
                    out_A[:, latent_dim_sensors:],
                )

                # Run through sensor B
                for t in training_times_B:
                    obs_B = true_y_dt_B[:, t, :]
                    out_B, h_B = rnn_encoder_B.forward(obs_B, h_B)
                # picking out mean and log variance for B sensor
                qz0_mean_B, qz0_logvar_B = (
                    out_B[:, :latent_dim_sensors],
                    out_B[:, latent_dim_sensors:],
                )

                qz0_mean_comb = torch.cat((qz0_mean_A, qz0_mean_B), dim=1)
                qz0_logvar_comb = torch.cat((qz0_logvar_A, qz0_logvar_B), dim=1)
                epsilon = torch.randn(qz0_mean_comb.size()).to(device)
                z0 = qz0_mean_comb

                # Sending z0 to odeint to integrate 'sample' timesteps forward
                # Permute just permutes the tensor.
                # First traject for vis.
                z0 = z0[0]
                pred_z = odeint(latent_node, z0, time_cont_A)
                pred_x = decoder(pred_z)
                
                # Predicting into the future with rnn forward!
                h_forward = rnn_encoder_forw.init_hidden_state().to(device)
                pred_times_forward = range(pred_true_AB.size()[1])
                pred_true_AB_dt_correct = pred_true_AB[:, args.samples : args.samples * 2 - 1, :]
                for t in pred_times_forward:
                    obs_forward = pred_true_AB[:, t, :]
                    out_forward, h_forward = rnn_encoder_forw.forward(obs_forward, h_forward)
                # picking out mean and log variance
                qz0_mean_forward, qz0_logvar_forward = (
                    out_forward[:, :latent_dim_total],
                    out_forward[:, latent_dim_total:],
                )

                epsilon = torch.randn(qz0_mean_forward.size()).to(device)
                #z0 = epsilon * torch.exp(0.5 * qz0_logvar_forward) + qz0_mean_forward
                z0 = qz0_mean_forward
                z0 = z0[0]
                # Creating time vector of timepoints we want to evaluate in the untrained half
                # Picks out the times from the first timepoint after we stopped the RNN forward
                # and then subtracts the time at which we stopped.
                t_forward_pred = torch.linspace(0,args.end_time,steps=100)

                pred_z = odeint(latent_node_fwd, z0, t_forward_pred)
                pred_x_forward = decoder_fwd(pred_z)



                plt.figure()
                plt.plot(
                    time_cont_A.cpu().detach().numpy(),
                    pred_x.cpu().detach().numpy(),
                    label="Predictions Back-RNN",
                )
                plt.plot(
                    time_cont_A.cpu().detach().numpy(),
                    true_y_cont_A.cpu().detach().numpy(),
                    label="Ground truth A",
                )
                plt.plot(
                    time_cont_B.cpu().detach().numpy(),
                    true_y_cont_B.cpu().detach().numpy(),
                    label="Ground truth B",
                )
                plt.plot((t_forward_pred+args.end_time).cpu().detach().numpy(), pred_x_forward.cpu().detach().numpy(), '-', label="RNN forward prediction")
                plt.axvspan(0, 20, facecolor="b", alpha=0.5)
                plt.legend()
                plt.draw()
                plt.savefig(
                    directory_name+"/ground_truth_with_pred_forward" + str(itr) + ".png",
                    dpi=500,
                )
                plt.close()
