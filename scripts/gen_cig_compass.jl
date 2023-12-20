using DrWatson
@quickactivate "WISE"
import Pkg; Pkg.instantiate()
using JLD2, JUDI, SegyIO, ImageGather
using ArgParse
using Statistics, Images
using FFTW
using LinearAlgebra
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

offsetrange = range(offset_start, stop=offset_end, length=n_offsets)

function ContJitter(l::Number, num::Int)
    #l = length, num = number of samples
    interval_width = l/num
    interval_center = range(interval_width/2, stop = l-interval_width/2, length=num)
    randomshift = interval_width .* rand(Float32, num) .- interval_width/2

    return interval_center .+ randomshift
end

f0 = 0.015f0
timeD = timeR = TD = 3200f0
dtD = 4f0
dtS = 4f0
nbl = 120

wavelet = ricker_wavelet(TD, dtS, f0)
wavelet = filter_data(wavelet, dtS; fmin=3f0, fmax=Inf)

d = (12.5f0, 12.5f0)
o = (0f0, 0f0)

# Plotting configs
background_type = "1d-special"
sim_name = "gen_ext-rtm-$(background_type)"
plot_path = joinpath(plotsdir(),sim_name)

data_path = datadir("m_train_compass.jld2")
if ~isfile(data_path)
    run(`wget https://www.dropbox.com/scl/fi/zq7p8xofbmfm7a2m0q8u6/'
        'm_train_compass.jld2 -q -O $file_path`)
end
m_train = JLD2.jldopen(data_path, "r")["m_train"];

n = (size(m_train, 1), size(m_train, 2))
# Setup model structure
nsrc = 64	# number of sources
model = Model(n, d, o, (1f0./m_train[:,:,1,1]).^2f0; nb=nbl)

nxrec = n[1]
xrec = range(0f0, stop=(n[1]-1)*d[1], length=nxrec)
yrec = 0f0 # WE have to set the y coordiante to zero (or any number) for 2D modeling
zrec = range(d[1], stop=d[1], length=nxrec)

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)

wb = 16
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range((wb-1)*d[1], stop=(wb-1)*d[1], length=nsrc))

function background_1d_average()
    m_mean = mean(m_train, dims=4)[:,:,1,1]
    wb = maximum(find_water_bottom(m_mean.-minimum(m_mean)))
    m0 = deepcopy(m_mean)
    m0[:,wb+1:end] .= 1f0./Float32.(imfilter(1f0./m_mean[:,wb+1:end], Kernel.gaussian(10)))
    return m0
end

function background_1d_gradient()
    m_1d_gradient = reshape(repeat(range(minimum(m_train), stop=maximum(m_train), length=n[2]), inner=n[1]), n)
    m0 = 1f0./Float32.(imfilter(1f0./m_1d_gradient, Kernel.gaussian(10)))
    return m0
end

if background_type == "1d-average"
    global m0 = background_1d_average()
elseif background_type == "1d-gradient"
    global m0 = background_1d_gradient()
else
    println("I do not know what background model to use")
end

m0_train = deepcopy(m_train)

snr = 12f0

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
    m_back = background_type == "1d-special" ? m0_train[:,:,1,i] : m0
    J = judiExtendedJacobian(F(1f0./m_back.^2f0), q, offsetrange)
    d_obs0 = F(1f0./m_back.^2f0) * q
    noise_ = deepcopy(d_obs)
    for l = 1:nsrc
        noise_.data[l] = randn(Float32, size(d_obs.data[l]))
        noise_.data[l] = real.(ifft(fft(noise_.data[l]).*fft(q.data[1])))
    end
    noise_ = noise_/norm(noise_) * norm(d_obs) * 10f0^(-snr/20f0)
    @time rtm = J' * (d_obs0 - (d_obs + noise_))
    save_dict = @strdict f0 dtD dtS nbl timeD timeR nsrc nxrec n d o q d_obs i rtm snr offset_start offset_end n_offsets
    @tagsave(
			joinpath(plot_path, savename(save_dict, "jld2"; digits=6)),
			save_dict;
			safe=true
		);
end