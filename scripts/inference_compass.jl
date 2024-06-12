#module load Julia/1.8/5; salloc -A rafael -t01:80:00 --gres=gpu:1 --mem-per-cpu=40G srun --pty julia 
#module load Julia/1.8/5; salloc -A rafael -t00:80:00 --partition=cpu --mem-per-cpu=20G srun --pty julia 
using DrWatson
@quickactivate "WISE"
import Pkg; Pkg.instantiate()
using JUDI
using ArgParse
using Random
Random.seed!(2023)
include("utils.jl")

# Parse command-line arguments
parsed_args = parse_commandline()
startidx = parsed_args["startidx"]
endidx = parsed_args["endidx"]
n_offsets = parsed_args["n_offsets"]
offset_start = parsed_args["offset_start"]
offset_end = parsed_args["offset_end"]
keep_offset_num = parsed_args["keep_offset_num"]

using InvertibleNetworks, Flux, UNet
using PyPlot,SlimPlotting
using LinearAlgebra, Random, Statistics
using ImageQualityIndexes
using BSON, JLD2
using Statistics, Images
using FFTW
using LinearAlgebra
using Random
using ImageGather
Random.seed!(2023)

# Plotting configs
background_type = "1d-average"
rtm_type = "ext-rtm"
sim_name = "inference-v-$(rtm_type)-$(background_type)"
plot_path = plotsdir(sim_name)

import DrWatson: _wsave
_wsave(s, fig::Figure) = fig.savefig(s, bbox_inches="tight", dpi=100)
PyPlot.rc("font", family="serif");

mkpath(datadir())

f0 = 0.015f0
timeD = timeR = TD = 3200f0
dtD = 4f0
dtS = 4f0
nbl = 120

wavelet = ricker_wavelet(TD, dtS, f0)
wavelet = filter_data(wavelet, dtS; fmin=3f0, fmax=Inf)

d = (12.5f0, 12.5f0)
o = (0f0, 0f0)
n = (512, 256)
# Setup model structure
nsrc = 64	# number of sources
nxrec = n[1]
snr = 12f0

# Training hyperparameters 
device = gpu

lr           = 8f-4
clipnorm_val = 3f0
noise_lev_x  = 0.1f0
noise_lev_init  = deepcopy(noise_lev_x)
noise_lev_y  = 0.0 

batch_size   = 8
n_epochs     = 200
num_post_samples = 64

save_every   = 10
plot_every   = 10
n_condmean   = 20

#### load example in the paper
#### or provide your own ones
# x is the unseen velocity model in size of 512 * 256
# y is the CIGs computed by migrating the observed data, in size of 512 * 256 * 51 * 1
# m0 is the migration velocity model
JLD2.@load datadir("wise-paper-example.jld2") x y m0 max_y

for z = 1:n[2]
	y[:,z,:,:] .*= z * d[2]
end

#normalize rtms
#max_y = quantile(abs.(vec(grad_train[:,:,:,1:300])),0.9999); computed by percentile of 300 CIGs in the training samples
y ./= max_y;

chan_obs   = n_offsets
chan_cond = 1
chan_target = 1

# Summary network parametrs
unet_lev = 4
unet = Chain(Unet(chan_obs, chan_cond, unet_lev)|> device);
trainmode!(unet, true); 
unet = FluxBlock(unet);

# Create conditional network
L = 3
K = 9 
n_hidden = 64
low = 0.5f0

Random.seed!(123);
cond_net = NetworkConditionalGlow(chan_target, chan_cond, n_hidden,  L, K;  split_scales=true, activation=SigmoidLayer(low=low,high=1.0f0)) |> device;
G        = SummarizedNet(cond_net, unet)

# Training logs 
net_path = datadir("trained-wise-cnf.bson")
net_link = "https://www.dropbox.com/scl/fi/o6x72s6e1chodnl8l79bd/trained-wise-cnf.bson?rlkey=n37wvo1gzbrhxwzmvfmamw5pt&dl=1"
if ~isfile(net_path)
    run(`wget -O $net_path $net_link`)
end

unet_lev = BSON.load(net_path)["unet_lev"];
n_hidden = BSON.load(net_path)["n_hidden"];
L = BSON.load(net_path)["L"];
K = BSON.load(net_path)["K"];

unet = Unet(chan_obs,1,unet_lev);
trainmode!(unet, true);
unet = FluxBlock(Chain(unet)) |> device;

cond_net = NetworkConditionalGlow(1, 1, n_hidden,  L, K;  freeze_conv=true, split_scales=true, activation=SigmoidLayer(low=0.5f0,high=1.0f0)) |> device;
G        = SummarizedNet(cond_net, unet)

Params = BSON.load(net_path)["Params"]; 
noise_lev_x = BSON.load(net_path)["noise_lev_x"]; 
set_params!(G,Params)

# Load in unet summary net
G.sum_net.model = BSON.load(net_path)["unet_model"]; 
G = G |> device;

#### Generate posterior samples X_gen

batch_size = 64
Y_train_latent_repeat = repeat(y |>cpu, 1, 1, 1, batch_size) |> device
Z_fix = randn(Float32, n[1],n[2],1,batch_size)|> device
_, Zy_fixed_train, _ = G.forward(Z_fix, Y_train_latent_repeat); #needs to set the proper sizes here
X_gen, Y_gen = G.inverse(Z_fix,Zy_fixed_train) |> cpu;
