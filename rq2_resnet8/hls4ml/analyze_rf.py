import re

src = open("firmware/parameters.h").read()
for m in re.finditer(r"struct (config\d+)[^{]*\{(.*?)\n\};", src, re.S):
    name, body = m.group(1), m.group(2)
    if "DenseResource" not in body:
        continue

    def g(k):
        mm = re.search(r"static const unsigned %s = (\d+);" % k, body)
        return int(mm.group(1)) if mm else None

    n_in = g("n_in")
    rf = g("reuse_factor")
    kern = re.search(r"DenseResource_(\w+)", body)
    kname = kern.group(1) if kern else "?"
    bad = (n_in is not None and rf is not None and rf > n_in and rf % n_in != 0)
    flag = "  <<< BAD rf_gt_nin (undefined template)" if bad else ""
    print("%-10s n_in=%-6s rf=%-4s kernel=%-18s%s" % (name, n_in, rf, kname, flag))
