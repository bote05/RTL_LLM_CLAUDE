"""
Verify the EXACT RTL unpack form (arithmetic-shift + sign-bit borrow add) used in
the Verilog, on a fixed-width 2's complement node, matches the Python signed-split.

RTL plan, S=18, node width W (signed):
  lo_signed = $signed(node[17:0])                 // sign-extend low 18 bits
  hi_signed = $signed(node[W-1:18]) + node[17]    // arith part + borrow (the LSB
                                                      of the truncated-away field's
                                                      MSB -> the sign of lo). +node[17]
                                                      because node>>>18 (arith) floors;
                                                      when lo<0 (node[17]=1) we add 1.
Check: node = hi_ref*2^18 + lo_ref  (2's complement in W bits). Recover hi_ref,lo_ref.
"""
S=18
def rtl_unpack(node, W):
    # node is python int in 2's complement range [-2^(W-1), 2^(W-1)-1]
    u = node & ((1<<W)-1)               # unsigned bit pattern
    lo_bits = u & ((1<<S)-1)            # node[17:0]
    lo = lo_bits - (1<<S) if (lo_bits >> (S-1)) & 1 else lo_bits   # $signed(node[17:0])
    hi_bits = (u >> S) & ((1<<(W-S))-1) # node[W-1:18] unsigned
    hi_arith = hi_bits - (1<<(W-S)) if (hi_bits >> (W-S-1)) & 1 else hi_bits  # $signed of that slice
    borrow = (u >> (S-1)) & 1           # node[17]
    hi = hi_arith + borrow
    return hi, lo

def py_split(acc):
    mask=(1<<S)-1; lo_u=acc&mask
    lo=lo_u-(1<<S) if lo_u>=(1<<(S-1)) else lo_u
    return (acc-lo)>>S, lo

import random
# packed4 node = sum of 4 packed products; width: packed_A(27b)*tap(8b)=35b prod, +2 =37b.
W=37
bad=0
for _ in range(500000):
    # build a real depth-4 packed sum
    acc=0
    wm=[]; wn=[]; a=[]
    for _k in range(4):
        wmv=random.randint(-128,127); wnv=random.randint(-128,127); av=random.randint(-128,127)
        A=(wnv<<S)+wmv
        acc += A*av
        wm.append(wmv); wn.append(wnv); a.append(av)
    # acc fits W bits signed?
    assert -(1<<(W-1)) <= acc <= (1<<(W-1))-1, acc
    hi,lo=rtl_unpack(acc,W)
    pyh,pyl=py_split(acc)
    LO_ref=sum(wm[k]*a[k] for k in range(4)); HI_ref=sum(wn[k]*a[k] for k in range(4))
    if (hi,lo)!=(HI_ref,LO_ref) or (hi,lo)!=(pyh,pyl): bad+=1
print(f"RTL-form unpack W={W} S={S}: bad={bad}/500000 (vs ref AND vs py_split)")

# extreme corners depth 4
ek=0
for wmv in(-128,127):
 for wnv in(-128,127):
  for av in(-128,127):
    acc=sum(((wnv<<S)+wmv)*av for _ in range(4))
    hi,lo=rtl_unpack(acc,W)
    if hi!=4*wnv*av or lo!=4*wmv*av: ek+=1
print(f"extreme depth-4: bad={ek}/8")
# Also confirm max node magnitude fits W=37
print(f"max |packed4| = |(127<<18+127)*127*4 type| -> A_max*128*4 = {(128*(1<<S)+128)*128*4} ; 2^36={1<<36}")
