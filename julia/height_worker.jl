# Height oracle worker: reads one request per line on stdin, writes one reply per line on stdout.
# Started (several copies in parallel) by rlmath.oracle.HeightOracle; nothing else should print to stdout.
#
#   MONOS n d                       -> the monomial order, as n*M space-separated exponents
#   H p n d cutoff c_1 ... c_M      -> "ok h" | "inf" (chain hit zero) | "past_cutoff" | "invalid" | "error <msg>"
#   S p n d c_1 ... c_M             -> "1" if V(f) ⊂ P^{n-1} is smooth over the algebraic closure, else "0"
#
# Coefficients are integers in 0..p-1, in the order returned by MONOS.

using Qfs
import Nemo
import Singular

# Monomials of degree d in n variables, exponent vectors in descending lex order.
# rlmath.polynomials.monomials must produce the same order; the client checks this at startup.
function monomials(n, d)
    n == 1 && return [[d]]
    out = Vector{Vector{Int}}()
    for a in d:-1:0, rest in monomials(n - 1, d - a)
        push!(out, [a; rest])
    end
    out
end

const MONOS = Dict{Tuple{Int,Int},Vector{Vector{Int}}}()
monos(n, d) = get!(() -> monomials(n, d), MONOS, (n, d))

const NEMO_RINGS = Dict{Tuple{Int,Int},Any}()
const SINGULAR_RINGS = Dict{Tuple{Int,Int},Any}()

function build(R, vars, coeffs, ms)
    f = zero(R)
    for (c, m) in zip(coeffs, ms)
        c == 0 && continue
        f += c * prod(vars .^ m)
    end
    f
end

function height(p, n, d, cutoff, coeffs)
    R, vars = get!(() -> Nemo.polynomial_ring(Nemo.GF(p), n), NEMO_RINGS, (p, n))
    f = build(R, vars, coeffs, monos(n, d))
    (f == 0 || n != d) && return "invalid"
    h = quasiFSplitHeight_CY_lift(p, f, cutoff)
    h == -1 && return "invalid"
    h == cutoff + 2 && return "inf"
    h == cutoff + 1 && return "past_cutoff"
    "ok $h"
end

# V(f) is smooth iff f and its partials have no common zero in projective space,
# i.e. the ideal (f, ∂f/∂x_1, ..., ∂f/∂x_n) has Krull dimension 0.
# f is included because when p | d the partials alone don't generate it (Euler's relation fails).
function smooth(p, n, d, coeffs)
    R, vars = get!(SINGULAR_RINGS, (p, n)) do
        Singular.polynomial_ring(Singular.Fp(p), ["x$i" for i in 1:n])
    end
    f = build(R, vars, coeffs, monos(n, d))
    f == 0 && return "0"
    I = Singular.Ideal(R, [f; [Singular.derivative(f, i) for i in 1:n]])
    Singular.dimension(Singular.std(I)) == 0 ? "1" : "0"
end

function handle(line)
    parts = split(line)
    cmd = parts[1]
    nums = parse.(Int, parts[2:end])
    if cmd == "MONOS"
        n, d = nums
        return join(reduce(vcat, monos(n, d)), " ")
    elseif cmd == "H"
        p, n, d, cutoff = nums[1:4]
        return height(p, n, d, cutoff, nums[5:end])
    elseif cmd == "S"
        p, n, d = nums[1:3]
        return smooth(p, n, d, nums[4:end])
    end
    "error unknown command $cmd"
end

function main()
    # Warm up so the first real request isn't slow.
    handle("H 3 4 4 10 " * join(fill(1, 35), " "))
    handle("S 3 4 4 " * join(fill(1, 35), " "))
    println("READY")
    flush(stdout)
    for line in eachline(stdin)
        isempty(strip(line)) && continue
        reply = try
            handle(line)
        catch e
            "error " * replace(sprint(showerror, e), '\n' => ' ')
        end
        println(reply)
        flush(stdout)
    end
end

main()
