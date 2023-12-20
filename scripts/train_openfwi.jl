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
Random.seed!(2023)

function posterior_sampler(G, y, x; device=gpu, num_samples=1, batch_size=16)
  size_x = size(x)
    # make samples from posterior for train sample 
  X_forward = randn(Float32, size_x[1:end-1]...,batch_size) |> device
    Y_train_latent_repeat = repeat(y |>cpu, 1, 1, 1, batch_size) |> device
    _, Zy_fixed_train, _ = G.forward(X_forward, Y_train_latent_repeat); #needs to set the proper sizes here

    X_post_train = zeros(Float32, size_x[1:end-1]...,num_samples)
    for i in 1:div(num_samples, batch_size)
      ZX_noise_i = randn(Float32, size_x[1:end-1]...,batch_size)|> device
      X_post_train[:,:,:, (i-1)*batch_size+1 : i*batch_size] = G.inverse(
          ZX_noise_i,
          Zy_fixed_train
        )[1] |> cpu;
  end
  X_post_train
end

function get_cm_l2_ssim(G, X_batch, Y_batch, X0_batch; device=gpu, num_samples=1)
		# needs to be towards target so that it generalizes accross iteration
	    num_test = size(Y_batch)[end]
	    l2_total = 0 
	    ssim_total = 0 
	    #get cm for each element in batch
	    for i in 1:num_test
	    	y   = Y_batch[:,:,:,i:i]
	    	x   = X_batch[:,:,:,i:i]

	    	X_post = posterior_sampler(G, y, x; device=device, num_samples=num_samples, batch_size=batch_size)
	    	x_hat =   mean(X_post; dims=4)[:,:,1,1]
	    	x_gt =  (x[:,:,1,1]) |> cpu
	    	ssim_total += assess_ssim(x_hat, x_gt)
				l2_total   += sqrt(mean((x_hat - x_gt).^2))
		end
	return l2_total / num_test, ssim_total / num_test
end

function get_loss(G, X_batch, Y_batch; device=gpu, batch_size=16)
	l2_total = 0 
	logdet_total = 0 
	num_batches = div(size(Y_batch)[end], batch_size)
	for i in 1:num_batches
		x_i = X_batch[:,:,:,(i-1)*batch_size+1 : i*batch_size] 
    	y_i = Y_batch[:,:,:,(i-1)*batch_size+1 : i*batch_size] 

    	x_i .+= noise_lev_x*randn(Float32, size(x_i)); 
    	y_i .+= noise_lev_y*randn(Float32, size(y_i)); 
    	Zx, Zy, lgdet = G.forward(x_i|> device, y_i|> device) |> cpu;
    	l2_total     += norm(Zx)^2 / (N*batch_size)
		logdet_total += lgdet / N
	end

	return l2_total / (num_batches), logdet_total / (num_batches)
end

# Plotting configs
background_type = "1d-special"
rtm_type = "ext-rtm"
sim_name = "cond-open-fwi-$(rtm_type)-$(background_type)"
plot_path = joinpath(plotsdir(),sim_name)

import DrWatson: _wsave
_wsave(s, fig::Figure) = fig.savefig(s, bbox_inches="tight", dpi=300)
PyPlot.rc("font", family="serif");

data_path = datadir("m_train_open_fwi.jld2")
if ~isfile(data_path)
    run(`wget https://www.dropbox.com/scl/fi/ylgus05wkhkvwchcxkjih/'
        'm_train_open_fwi.jld2 -q -O $file_path`)
end
m_train = JLD2.jldopen(data_path, "r")["m_train"];

f0 = 0.015f0
timeD = timeR = TD = 1000f0
dtD = 1f0
dtS = 1f0
nbl = 120
ntS = div(TD, dtS) + 1
wavelet_unfiltered = ricker_wavelet(TD, dtS, f0)
wavelet = filter_data(wavelet_unfiltered, dtS; fmin=3f0, fmax=Inf)

d = (10f0, 10f0)
o = (0f0, 0f0)
n = (64, 64)
# Setup model structure
nsrc = 16	# number of sources
nxrec = n[1]
snr = 12f0

m0_train = nothing

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
plot_every   = 1
n_condmean   = 20

n_tot_sample = size(m_train)[end]

grad_train = zeros(Float32, size(m_train, 1), size(m_train, 2), keep_offset_num, size(m_train, 4))
keep_offset_idx = div(n_offsets,2)+1-div(keep_offset_num, 2):div(n_offsets,2)+1+div(keep_offset_num, 2)
for i = 1:n_tot_sample
	if background_type == "1d-average"
		misc_dict = @strdict f0 dtD dtS nbl timeD timeR nsrc nxrec n d o i snr offset_start offset_end n_offsets
	else
		misc_dict = @strdict background_type f0 dtD dtS nbl timeD timeR nsrc nxrec n d o i snr offset_start offset_end n_offsets
	end
    grad_train[:,:,:,i] = permutedims(JLD2.jldopen(joinpath(joinpath(plotsdir() "openfwi", "gen-ext-rtm"), savename(misc_dict, "jld2"; digits=6)), "r")["rtm"][keep_offset_idx, :, :], [2, 3, 1])
    for x = 1:n[1]
        for z = 1:n[2]
            grad_train[x,z,:,i] .*= z * d[2]
        end
    end
end

m_back = 1f0./mean(1f0./m_train[:,:,:,1:1040], dims=4)[:,:,1,1]
for i = 1:size(m_back, 1)
    m_back[i,:] = mean(m_back, dims=1)
end
m_back = 1f0./Float32.(imfilter(1f0./m_back, Kernel.gaussian(1)))
m0_train = deepcopy(m_train)
for i = 1:size(m0_train)[end]
    m0_train[:,:,1,i] = m_back
end

#normalize rtms
max_y = quantile(abs.(vec(grad_train[:,:,:,1:300])),0.9999);
grad_train ./= max_y;

num_train = 2800
target_train = m_train[:,:,:,1:num_train];
X0_train     = m0_train[:,:,:,1:num_train];
Y_train      = grad_train[:,:,:,1:num_train];

target_test = m_train[:,:,:,(num_train+50):end];
X0_test     = m0_train[:,:,:,(num_train+50):end];
Y_test      = grad_train[:,:,:,(num_train+50):end];

n_x, n_y, chan_target, n_train = size(target_train)
n_train = size(target_train)[end]
N = n_x*n_y*chan_target
chan_obs   = size(Y_train)[end-1]
chan_cond = 1

X_train  = target_train
X_test   = target_test  

vmax_v = maximum(X_train)
vmin_v = minimum(X_train)

n_batches    = cld(n_train, batch_size)-1
n_train_safe = batch_size*n_batches

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

# Optimizer
opt = Flux.Optimiser(ClipNorm(clipnorm_val), ADAM(lr))

# Training logs 
loss      = []; logdet_train = []; ssim      = []; l2_cm      = [];
loss_test = []; logdet_test  = []; ssim_test = []; l2_cm_test = [];

noise_lev_x_min =  1f-3
noise_decay_per_epochs = div(n_epochs-50, Int(floor(log(noise_lev_x_min/noise_lev_init)/log(1f0/1.2f0))+1))

for e=1:n_epochs # epoch loop
	idx_e = reshape(randperm(n_train)[1:n_train_safe], batch_size, n_batches) 

	if (e >= 30) && (e <= n_epochs-20) && (mod(e,noise_decay_per_epochs) == 0)
        global noise_lev_x /= 1.2f0
        global noise_lev_x = max(noise_lev_x, noise_lev_x_min)
    end

	for b = 1:n_batches # batch loop
    X = X_train[:, :, :, idx_e[:,b]];
    Y = Y_train[:, :, :, idx_e[:,b]];

		for i in 1:batch_size #quick data augmentation to prevent overfitting
	  	if rand() > 0.5
    		X[:,:,:,i:i] = X[end:-1:1,:,:,i:i]
    		Y[:,:,:,i:i] = Y[end:-1:1,:,:,i:i]
	  	end
    end

    X .+= noise_lev_x*randn(Float32, size(X)); #noises not related to inverse problem 
    Y .+= noise_lev_y*randn(Float32, size(Y))
    Y = Y |> device; 
    
    Zx, Zy, lgdet = G.forward(X |> device, Y)

    # Loss function is l2 norm 
    append!(loss, norm(Zx)^2 / (N*batch_size))  # normalize by image size and batch size
    append!(logdet_train, -lgdet / N) # logdet is internally normalized by batch size

    # Set gradients of flow and summary network
    dx, x, dy = G.backward(Zx / batch_size, Zx, Zy; Y_save = Y)

    for p in get_params(G) 
      Flux.update!(opt,p.data,p.grad)
    end; clear_grad!(G)

    print("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, 
          "; f l2 = ",  loss[end], 
          "; lgdet = ", logdet_train[end], "; f = ", loss[end] + logdet_train[end], "\n")
    Base.flush(Base.stdout)
	end
  
    if(mod(e,plot_every)==0) 
    	#get loss of training objective on test set corresponds to mutual information between summary statistic and x
	    @time l2_test_val, lgdet_test_val  = get_loss(G, X_test, Y_test; device=device, batch_size=batch_size)
	    append!(logdet_test, -lgdet_test_val)
	    append!(loss_test, l2_test_val)

	    # get conditional mean metrics over training batch  
	    @time cm_l2_train, cm_ssim_train = get_cm_l2_ssim(G, X_train[:,:,:,1:n_condmean], Y_train[:,:,:,1:n_condmean], X0_train[:,:,:,1:n_condmean]; device=device, num_samples=num_post_samples)
	    append!(ssim, cm_ssim_train)
	    append!(l2_cm, cm_l2_train)

	    # get conditional mean metrics over testing batch  
	    @time cm_l2_test, cm_ssim_test  = get_cm_l2_ssim(G, X_test[:,:,:,1:n_condmean], Y_test[:,:,:,1:n_condmean], X0_test[:,:,:,1:n_condmean]; device=device, num_samples=num_post_samples)
	    append!(ssim_test, cm_ssim_test)
	    append!(l2_cm_test, cm_l2_test)

	    for (test_x, test_y, test_x0, file_str) in [[X_train,Y_train, X0_train, "train"], [X_test, Y_test, X0_test, "test"]]
		    num_cols = 7
	    	plots_len = 2
	    	all_sampls = size(test_x)[end]-1
		    fig = figure(figsize=(25, 5)); 
		    for (i,ind) in enumerate((2:div(all_sampls,3):all_sampls)[1:plots_len])
		    	x0 = test_x0[:,:,1,ind] 
			    x = test_x[:,:,:,ind:ind] 
			    y = test_y[:,:,:,ind:ind]
			    y .+= noise_lev_y*randn(Float32, size(y));

			    # make samples from posterior for train sample 
			   	X_post = posterior_sampler(G,  y, x; device=device, num_samples=num_post_samples,batch_size=batch_size)|> cpu
			    X_post_mean = mean(X_post,dims=4)
			    X_post_std  = std(X_post, dims=4)

		    	x_hat =  X_post_mean[:,:,1,1]
		    	x_gt =   x[:,:,1,1]
			    error_mean = abs.(x_hat-x_gt)

			    ssim_i = round(assess_ssim(x_hat, x_gt),digits=2)
		    	rmse_i = round(sqrt(mean(error_mean.^2)),digits=4)

				y_plot = y[:,:,div(keep_offset_num, 2)+1, 1]
			    a = quantile(abs.(vec(y_plot)), 98/100)

			    subplot(plots_len,num_cols,(i-1)*num_cols+1); imshow(y_plot', vmin=-a,vmax=a,interpolation="none", cmap="gray")
					axis("off"); title("Migration");#colorbar(fraction=0.046, pad=0.04);

			    subplot(plots_len,num_cols,(i-1)*num_cols+2); imshow(X_post[:,:,1,1]', vmin=vmin_v,vmax=vmax_v, interpolation="none", cmap="cet_rainbow4")
					axis("off"); title("Posterior sample") #colorbar(fraction=0.046, pad=0.04);
				
					subplot(plots_len,num_cols,(i-1)*num_cols+3); imshow(X_post[:,:,1,2]', vmin=vmin_v,vmax=vmax_v, interpolation="none", cmap="cet_rainbow4")
					axis("off");title("Posterior sample")  #colorbar(fraction=0.046, pad=0.04);title("Posterior sample")

					subplot(plots_len,num_cols,(i-1)*num_cols+4); imshow(x_gt',  vmin=vmin_v,vmax=vmax_v,  interpolation="none", cmap="cet_rainbow4")
					axis("off"); title(L"Reference $\mathbf{x^{*}}$") ; #colorbar(fraction=0.046, pad=0.04)

					subplot(plots_len,num_cols,(i-1)*num_cols+5); imshow(x_hat' ,  vmin=vmin_v,vmax=vmax_v,  interpolation="none", cmap="cet_rainbow4")
					axis("off"); title("Posterior mean | SSIM="*string(ssim_i)) ; #colorbar(fraction=0.046, pad=0.04)

					subplot(plots_len,num_cols,(i-1)*num_cols+6); imshow(error_mean' , vmin=0,vmax=nothing, interpolation="none", cmap="magma")
					axis("off");title("Error | RMSE="*string(rmse_i)) ;# cb = colorbar(fraction=0.046, pad=0.04)

					subplot(plots_len,num_cols,(i-1)*num_cols+7); imshow(X_post_std[:,:,1,1]' , vmin=0,vmax=nothing,interpolation="none", cmap="magma")
					axis("off"); title("Posterior variance") ;#cb =colorbar(fraction=0.046, pad=0.04)
				end
				tight_layout()
			  fig_name = @strdict num_train background_type chan_obs noise_lev_x noise_lev_init n_train e offset_start offset_end n_offsets keep_offset_num
			  safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_"*file_str*".png"), fig); close(fig)
			end
		
	    ############# Training metric logs
		if e != plot_every
			sum_train = loss + logdet_train
			sum_test = loss_test + logdet_test

			fig = figure("training logs ", figsize=(10,12))
			subplot(5,1,1); title("L2 Term: train="*string(loss[end])*" test="*string(loss_test[end]))
			plot(range(0f0, 1f0, length=length(loss)), loss, label="train");
			plot(range(0f0, 1f0, length=length(loss_test)),loss_test, label="test"); 
			axhline(y=1,color="red",linestyle="--",label="Normal Noise")
			ylim(bottom=0.,top=1.5)
			xlabel("Parameter Update"); legend();

			subplot(5,1,2); title("Logdet Term: train="*string(logdet_train[end])*" test="*string(logdet_test[end]))
			plot(range(0f0, 1f0, length=length(logdet_train)),logdet_train);
			plot(range(0f0, 1f0, length=length(logdet_test)),logdet_test);
			xlabel("Parameter Update") ;

			subplot(5,1,3); title("Total Objective: train="*string(sum_train[end])*" test="*string(sum_test[end]))
			plot(range(0f0, 1f0, length=length(sum_train)),sum_train); 
			plot(range(0f0, 1f0, length=length(sum_test)),sum_test); 
			xlabel("Parameter Update") ;

			subplot(5,1,4); title("SSIM train=$(ssim[end]) test=$(ssim_test[end])")
	    plot(range(0f0, 1f0, length=length(ssim)),ssim); 
	    plot(range(0f0, 1f0, length=length(ssim_test)),ssim_test); 
	    xlabel("Parameter Update") 

	    subplot(5,1,5); title("RMSE train=$(l2_cm[end]) test=$(l2_cm_test[end])")
	    plot(range(0f0, 1f0, length=length(l2_cm)),l2_cm); 
	    plot(range(0f0, 1f0, length=length(l2_cm_test)),l2_cm_test); 
	    xlabel("Parameter Update") 

			tight_layout()
			fig_name = @strdict num_train background_type chan_obs noise_lev_x noise_lev_init n_train e offset_start offset_end n_offsets keep_offset_num
			safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_log.png"), fig); close(fig)
		end

	end

	if(mod(e,save_every)==0) 
		unet_model = G.sum_net.model|> cpu;
		G_save = deepcopy(G);
		reset!(G_save.sum_net); # clear params to not save twice
		Params = get_params(G_save) |> cpu;
		save_dict = @strdict num_train background_type chan_obs unet_lev unet_model n_train e noise_lev_x noise_lev_init lr n_hidden L K Params loss logdet_train l2_cm ssim loss_test logdet_test l2_cm_test ssim_test batch_size offset_start offset_end n_offsets keep_offset_num; 
		@tagsave(
			joinpath(plotsdir("savednets_openfwi"), savename(save_dict, "bson"; digits=6)),
			save_dict;
			safe=true
		);
	end
end