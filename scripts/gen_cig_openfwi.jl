using DrWatson
@quickactivate "WISE"
import Pkg; Pkg.instantiate()
using JLD2, JUDI, SegyIO
using Random
using ArgParse
using Statistics
using Images
using ImageGather
using LinearAlgebra
Random.seed!(2023)
function ContJitter(l::Number, num::Int)
    #l = length, num = number of samples
    interval_width = l/num
    interval_center = range(interval_width/2, stop = l-interval_width/2, length=num)
    randomshift = interval_width .* rand(Float32, num) .- interval_width/2

    return interval_center .+ randomshift
end
include("utils.jl")
# Parse command-line arguments
parsed_args = parse_commandline()
startidx = parsed_args["startidx"]
endidx = parsed_args["endidx"]
n_offsets = parsed_args["n_offsets"]
offset_start = parsed_args["offset_start"]
offset_end = parsed_args["offset_end"]

offsetrange = range(offset_start, stop=offset_end, length=n_offsets)

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
# Plotting configs
sim_name = "openfwi"
exp_name = "gen-data"
exp_name_rtm = "gen-ext-rtm"
plot_path = joinpath(plotsdir(),sim_name,exp_name)
plot_path_rtm = joinpath(plotsdir(),sim_name,exp_name_rtm)
data_path = datadir("m_train_open_fwi.jld2")
if ~isfile(data_path)
    run(`wget https://www.dropbox.com/scl/fi/ylgus05wkhkvwchcxkjih/'
        'm_train_open_fwi.jld2 -q -O $data_path`)
end
m_train = JLD2.jldopen(data_path, "r")["m_train"];

background_type = "1d-special"

# Setup model structure
nsrc = 16	# number of sources
model = Model(n, d, o, (1f0./m_train[:,:,1,1]).^2f0; nb=nbl)

nxrec = n[1]
xrec = range(0f0, stop=(n[1]-1)*d[1], length=nxrec)
yrec = 0f0 # WE have to set the y coordiante to zero (or any number) for 2D modeling
zrec = range(d[1], stop=d[1], length=nxrec)

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)

ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(d[1], stop=d[1], length=nsrc))
snr = 12f0

m_back = 1f0./mean(1f0./m_train[:,:,:,1:1040], dims=4)[:,:,1,1]
for i = 1:size(m_back, 1)
    m_back[i,:] = mean(m_back, dims=1)
end
m_back = 1f0./Float32.(imfilter(1f0./m_back, Kernel.gaussian(1)))

# Setup operators
for i = startidx:endidx
    Base.flush(Base.stdout)
    @info "sample $i out of $(size(m_train)[end]) samples"
    # Set up source structure
    xsrc = convertToCell(ContJitter((n[1]-1)*d[1], nsrc))
    srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)
    q = judiVector(srcGeometry, wavelet)
    F = judiModeling(model, srcGeometry, recGeometry)
    @time d_obs = F(1f0./m_train[:,:,1,i].^2f0) * q
    m_back_now = (background_type == "1d-average") ? m_back : (1f0./Float32.(imfilter(repeat(mean(1f0./m_train[:,:,1,i], dims=1), 64), Kernel.gaussian(1))))
    J = judiExtendedJacobian(F(1f0./m_back_now.^2f0), q, offsetrange)
    noise_ = deepcopy(d_obs)
    for l = 1:nsrc
        noise_.data[l] = randn(Float32, size(d_obs.data[l]))
        noise_.data[l] = real.(ifft(fft(noise_.data[l]).*fft(q.data[1])))
    end
    noise_ = noise_/norm(noise_) * norm(d_obs) * 10f0^(-snr/20f0)
    @time rtm = J' * (d_obs + noise_)
    save_dict_rtm = @strdict background_type f0 dtD dtS nbl timeD timeR nsrc nxrec n d o m_back_now q d_obs i rtm snr offset_start offset_end n_offsets
    @tagsave(
            joinpath(plot_path_rtm, savename(save_dict_rtm, "jld2"; digits=6)),
            save_dict_rtm;
            safe=true
            );
end
