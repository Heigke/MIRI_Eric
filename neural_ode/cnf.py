""" Attempt to do some CNF """
import torchdiffeq
from torchdiffeq import odeint as odeint
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import datetime

# First off generate data
def generate_unkn_dist():
    # Several simple distributions spread out
    x_unkn_dist = []
    y_unkn_dist = []
    nbr_of_dist = 5
    angle_inc = 2 * np.pi / nbr_of_dist
    points_per_dist = 2000
    for i in range(nbr_of_dist):

        mean = [
            0.5 * (np.cos(i * angle_inc + np.pi / 4)),
            0.5 * (np.sin(i * angle_inc + np.pi / 4)),
        ]
        cov = [[0.025, 0], [0, 0.025]]
        x, y = np.random.multivariate_normal(mean, cov, points_per_dist).T
        x_unkn_dist.append(x)
        y_unkn_dist.append(y)

    x_unkn_dist = np.asarray(x_unkn_dist)
    y_unkn_dist = np.asarray(y_unkn_dist)
    x_unkn_dist = np.reshape(x_unkn_dist, (nbr_of_dist * points_per_dist))
    y_unkn_dist = np.reshape(y_unkn_dist, (nbr_of_dist * points_per_dist))
    r = torch.randint(0, x_unkn_dist.size, (x_unkn_dist.size,))
    x_unkn_dist[r]
    y_unkn_dist[r]
    return x_unkn_dist, y_unkn_dist


# Single gaussian distribution
def generate_known_dist(nbr_points):
    mean = [0.0, 0.0]
    cov = [[0.1, 0.0], [0.0, 0.1]]
    x, y = np.random.multivariate_normal(mean, cov, nbr_points).T
    return x, y, mean, cov


# dZ/dt = f = this function which transforms points in respect to depth/time
# Inspired by researchers who add bias and gate dependent on time at each layer.
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        dim_in = 2
        dim_out = 60
        self.in_layer = nn.Linear(dim_in, dim_out)

        self._hyper_bias_in = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate_in = nn.Linear(1, dim_out)

        self.middle_layer = nn.Linear(dim_out, dim_out)
        self._hyper_bias_middle = nn.Linear(1, dim_out)
        self._hyper_gate_middle = nn.Linear(1, dim_out)

        self.out_layer = nn.Linear(dim_out, dim_in)
        self._hyper_bias_out = nn.Linear(1, dim_in, bias=False)
        self._hyper_gate_out = nn.Linear(1, dim_in)
        self.activation = nn.Tanh()

        self.net = nn.Sequential(
            nn.Linear(3, 20),
            nn.Tanh(),
            nn.Linear(20, 50),
            nn.Tanh(),
            nn.Linear(50, 60),
            nn.Tanh(),
            nn.Linear(60, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward2(self, t, y):
        y = y.float()
        t = t.expand((y.size(0), 1))
        input = torch.cat((y, t), dim=1)
        return self.net(input)

    def forward(self, t, y):
        y = y.float()
        t = t.view(1, 1)
        first_layer_seq = self.in_layer(y) * torch.sigmoid(
            self._hyper_gate_in(t.view(1, 1))
        ) + self._hyper_bias_in(t.view(1, 1))
        first_layer_seq = self.activation(first_layer_seq)

        middle_layer_seq = self.middle_layer(first_layer_seq) * torch.sigmoid(
            self._hyper_gate_middle(t.view(1, 1))
        ) + self._hyper_bias_middle(t.view(1, 1))

        middle_layer_seq = self.middle_layer(middle_layer_seq) * torch.sigmoid(
            self._hyper_gate_middle(t.view(1, 1))
        ) + self._hyper_bias_middle(t.view(1, 1))

        second_layer_seq = self.out_layer(middle_layer_seq) * torch.sigmoid(
            self._hyper_gate_out(t.view(1, 1))
        ) + self._hyper_bias_out(t.view(1, 1))
        second_layer_seq = self.activation(second_layer_seq)
        return second_layer_seq


# The integrand is contained in this function, see FFJORD eq (4)
class faugClass(nn.Module):
    def __init__(self, nbatches=1):
        super(faugClass, self).__init__()
        self.eps = torch.randn((nbatches, 2), requires_grad=False)
        self.ode_func = ODEFunc()
        self.nbatches = nbatches

    def update_eps(self, nbatches):
        self.eps = torch.randn((nbatches, 2), requires_grad=False)

    def return_ode_func(self):
        return self.ode_func

    def update_batch_size(self, nbatches):
        self.nbatches = nbatches

    # Both the Hutchinsons trace estimator which corresponds to the d lg(p(z(t)))/dt
    # and the dz(t)/dt=f(t) is calculated here
    def faug_integrand(self, t, input_z_pz, ode_func, eps, nbatches):
        z = input_z_pz[0]
        e_dzdx = torch.autograd.grad(ode_func.forward(t, z), z, eps, create_graph=True)[
            0
        ]
        e_dzdx_e = e_dzdx * eps
        approx_tr_dzdx = e_dzdx_e.view(z.shape[0], -1).sum(dim=1)
        approx_tr_dzdx = torch.reshape(approx_tr_dzdx, [nbatches, 1])
        f_t = ode_func.forward(t, z)
        return (f_t, approx_tr_dzdx)

    def forward(self, t, input_z_pz):
        return self.faug_integrand(
            t, input_z_pz, self.ode_func, self.eps, self.nbatches
        )


# Calculate the probability density function value of a gaussian for a specific x.
def log_multivariate_normal_pdf(mu, sigma, x, nbatches):
    mu = np.asarray(mu)
    mu = torch.from_numpy(mu).float()
    mu = mu.unsqueeze(-1)
    sigma = np.asarray(sigma)
    sigma = torch.from_numpy(sigma)
    # Make sure that the right axises are transposed, first index batches
    x_mu_T = torch.transpose(x - mu, dim0=1, dim1=2)
    inv_sigma = torch.inverse(sigma).float()
    inv_sigma = inv_sigma.expand(nbatches, 2, 2)
    x_mu_T_inv_sigma = torch.bmm(x_mu_T, inv_sigma)
    x_mu = x - mu
    x_mu = x_mu.expand(nbatches, 2, 1)
    exponent = torch.bmm(x_mu_T_inv_sigma, x_mu)
    denominator = np.sqrt((2 * np.pi) ** 2 * torch.det(sigma))
    pdf_x = torch.exp(-1 / 2 * exponent) / denominator

    return torch.log(pdf_x)


# Instead try a more complex distribution
def generate_logo_samples():
    # Visualizing sentian logo
    from PIL import Image

    img = np.array(Image.open("maple_leaf.jpg").convert("L"))
    h, w = img.shape
    xx = np.linspace(-2, 2, w)
    yy = np.linspace(-2, 2, h)
    xx, yy = np.meshgrid(xx, yy)
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)

    means = np.concatenate([xx, yy], 1)
    img = img.max() - img
    probs = img.reshape(-1) / img.sum()

    std = np.array([8 / w / 2, 8 / h / 2])

    batch_size = 20000
    inds = np.random.choice(int(probs.shape[0]), int(batch_size), p=probs)
    m = means[inds]
    samples = np.random.randn(*m.shape) * std + m

    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.hist2d(samples[:, 0], samples[:, 1], bins=100)
    plt.show()
    # ******************
    return samples[:, 0], samples[:, 1]


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


if __name__ == "__main__":
    # Creating loss meter
    loss_meter = RunningAverageMeter(0.97)
    # Generating distributions
    #x_unkn, y_unkn = generate_logo_samples()
    x_unkn, y_unkn = generate_unkn_dist()
    x_known, y_known, mean_known, cov_known = generate_known_dist(x_unkn.size)
    # Batches
    nbatches = x_unkn.size
    # Create faug instance
    f_aug = faugClass(nbatches)
    # Create parameters & optimizer
    params = f_aug.return_ode_func().parameters()
    optimizer = optim.Adam(params, lr=0.0002)
    # Vis
    visualize = True

    # Create folder name
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
    directory_name = "cnf_" + str_current_day
    # Plot data generated
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes[0, 0].set_title("Sampled Distribution")
    axes[0, 0].hist2d(x_unkn[:], y_unkn[:], bins=100)
    axes[0, 1].set_title("Desired Distribution")
    axes[0, 1].hist2d(x_known[:], y_known[:], bins=100)
    plt.show(block=True)
    niters = 25000

    for i in range(niters):
        # Update the hutchinsons trace
        f_aug.update_eps(nbatches)
        f_aug.update_batch_size(nbatches)
        # rnd_ind = np.random.randint(0, x_unkn.size - nbatches, 1)[0]
        rnd_ind = 0
        # Pick random points
        # Or pick all points so that p(z(t1)) = 1 makes sense! :O
        x = np.array(
            [x_unkn[rnd_ind : rnd_ind + nbatches], y_unkn[rnd_ind : rnd_ind + nbatches]]
        )

        x = torch.from_numpy(x).float()
        x = torch.autograd.Variable(x, requires_grad=True)
        x = torch.transpose(x, 0, 1)
        # Initial value for the trace integral
        zero = torch.zeros([nbatches]).float()
        zero = torch.autograd.Variable(zero, requires_grad=True)
        zero = zero.unsqueeze(-1)
        # We go backwards in time
        t = torch.tensor([1.0, 0.5, 0.0]).float()

        z, delta_p = odeint(f_aug, (x, zero), t, method="rk4")
        # Pick out the last time point
        z = z[-1, :, :]
        z = z.unsqueeze(2)

        lg_pz0 = log_multivariate_normal_pdf(mean_known, cov_known, z, nbatches)
        delta_p = delta_p[-1, :].unsqueeze(1)
        # Basically minimize lg(p(x)) i.o.w maximize probability for each x point?
        loss_term = lg_pz0 + (delta_p)
        loss_term = -torch.mean(loss_term)
        loss_term.backward()
        optimizer.step()
        loss_meter.update(loss_term.item())
        if i % 50 == 0:
            print("*********iteration" + str(i) + "**************")
            print("Total Loss: " + str(loss_term))
            print("Iter {:04d} | Running Loss {:.6f}".format(i, loss_term.item()))
            print("lg_pzo: " + str(torch.mean(lg_pz0)))
            print("delta_p: " + str(torch.mean(torch.abs(delta_p))))
            print("******************************")

        # Put in gaussian distribution and integrate forward in time to visualize
        if visualize and i % 100 == 0:

            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
            nbr_points = x_unkn.size
            zero = torch.zeros([nbr_points]).float()
            zero = zero.unsqueeze(-1)
            x = np.array([x_known[0:nbr_points], y_known[0:nbr_points]])
            x = torch.from_numpy(x).float()
            x = torch.autograd.Variable(x, requires_grad=True)
            x = torch.transpose(x, 0, 1)
            f_aug.update_eps(nbr_points)
            f_aug.update_batch_size(nbr_points)
            zero = torch.zeros([nbr_points]).float()
            zero = torch.autograd.Variable(zero, requires_grad=True)
            zero = zero.unsqueeze(-1)
            t = (
                torch.tensor(
                    [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
                ).float()
                / 3.0
            )
            z, delta_p = odeint(f_aug, (x, zero), t, method="rk4")
            z_all = z[:, :, :]
            z = z[-1, :, :]
            z = z.detach().numpy()
            z_all = z_all.detach().numpy()
            fig, axes = plt.subplots(nrows=2, ncols=2)
            axes[0, 0].set_title("Sampled Distribution")
            axes[0, 0].hist2d(x_unkn[:], y_unkn[:], bins=100)
            axes[0, 1].set_title("Desired Distribution")
            axes[0, 1].hist2d(x_known[:], y_known[:], bins=100)
            axes[1, 0].hist2d(z[:, 0], z[:, 1], bins=100)
            axes[1, 0].set_title("Transformed desired dist.")
            print("Saving:" + directory_name + "/cnf" + str(i) + ".png")
            plt.savefig(directory_name + "/cnf" + str(i) + ".png", dpi=500)
            plt.close()
            for i in range(11):
                fig, axes = plt.subplots(nrows=1, ncols=1)
                axes.set_title("Animation of transformation")
                axes.hist2d(z_all[i, :, 0], z_all[i, :, 1], bins=100)
                plt.savefig(directory_name + "/cnf_anim" + str(i) + ".png", dpi=500)
                plt.close()
            print("Visualization done and saved")
            print("Saving model")
            torch.save(f_aug.state_dict(), directory_name + "/model.pt")
            print("Model saved")
            print("Saving optimizer")
            torch.save(optimizer.state_dict(), directory_name + "/optimizer.pt")

