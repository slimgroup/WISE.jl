function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--startidx"
            help = "Start index"
            arg_type = Int
            default = 1
        "--endidx"
            help = "End index"
            arg_type = Int
            default = 1000
        "--n_offsets"
            help = "num of offsets"
            arg_type = Int
            default = 51
        "--offset_start"
            help = "start of offset"
            arg_type = Float32
            default = -500f0
        "--offset_end"
            help = "end of offset"
            arg_type = Float32
            default = 500f0
        "--keep_offset_num"
            help = "keep how many offset during training"
            arg_type = Int
            default = 51
    end
    return parse_args(s)
end

function ContJitter(l::Number, num::Int)
    #l = length, num = number of samples
    interval_width = l/num
    interval_center = range(interval_width/2, stop = l-interval_width/2, length=num)
    randomshift = interval_width .* rand(Float32, num) .- interval_width/2

    return interval_center .+ randomshift
end