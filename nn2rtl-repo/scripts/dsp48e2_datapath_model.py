"""
Bit-accurate model of the DSP48E2 datapath for OPMODE=9'b000000101, ALUMODE=4'b0000,
USE_MULT=MULTIPLY, INMODE=0, no pattern detect, no pre-adder.

DSP48E2 ALU final-add (from UG579 / the unisim model):
  X mux (OPMODE[1:0]):
    00 -> 0
    01 -> M[?]   (multiplier partial-product low; together with Y forms the 43/45b product)
    10 -> P
    11 -> A:B (concatenation)
  Y mux (OPMODE[3:2]):
    00 -> 0
    01 -> M (multiplier partial-product high)
    10 -> 48'hFFFFFFFFFFFF
    11 -> C
  Z mux (OPMODE[6:4]):
    000 -> 0
    ...
  With OPMODE[3:0]=0101 the X and Y muxes jointly select the 45-bit signed product
  of the 27x18 multiplier (this is the canonical "multiply" selection). Z=000 -> 0.
  ALUMODE=0000 -> P = Z + (X + Y + CARRYIN) = product.

So P = sign_extend_to_48( A_signed[26:0] * B_signed[17:0] ), CARRYIN=0.
We verify P[34:0] == (w_n*tap)<<18 + (w_m*tap) where A=(w_n<<18)+w_m, B=tap.
"""
import random
S = 18
def dsp48e2_multiply(A27, B18):
    # 27-bit signed A, 18-bit signed B -> 45-bit signed product, placed in 48-bit P.
    # (OPMODE 0101 / Z=0 / ALUMODE 0 / CARRYIN 0)
    P = A27 * B18                      # full signed product
    # P register is 48-bit; the multiplier output is 45-bit sign-extended into 48.
    Pmask = P & ((1<<48)-1)
    return Pmask

def sx(v, bits):
    m = 1 << (bits-1)
    return (v ^ m) - m

bad = 0
N = 1_000_000
random.seed(7)
corners = [-128,127]
cases = [(wm,wn,t) for wm in corners for wn in corners for t in corners]
cases += [(random.randint(-128,127), random.randint(-128,127), random.randint(-128,127)) for _ in range(N)]
for wm,wn,t in cases:
    A = (wn << S) + wm                 # signed 27-bit packed
    assert -(1<<26) <= A <= (1<<26)-1, A   # fits 27-bit signed
    B = t                              # signed 18-bit (8-bit value sign-extended is identical)
    Praw = dsp48e2_multiply(A, B)
    p35 = sx(Praw & ((1<<35)-1), 35)
    ref = (wn*t)*(1<<S) + (wm*t)
    if p35 != ref:
        if bad < 5: print("BAD", wm,wn,t,"p35",p35,"ref",ref)
        bad += 1
print(f"DSP48E2 datapath model (OPMODE=000000101, ALUMODE=0): bad={bad}/{len(cases)}")
# Also confirm A fits 27-bit signed at the extreme:
amax = (127<<S)+127; amin=(-128<<S)+(-128)
print(f"A range [{amin},{amax}]  27-bit signed range [{-(1<<26)},{(1<<26)-1}]  fits={amin>=-(1<<26) and amax<=(1<<26)-1}")
