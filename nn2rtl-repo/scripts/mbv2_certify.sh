#!/usr/bin/env bash
# Certify a MobileNetV2 route from its AUTHORITATIVE post-route timing rpt.
# Usage: bash scripts/mbv2_certify.sh <timing_rpt_path>
# Prints: clock period, WNS, WHS, worst-path, Fmax (MHz), fps (= Fmax/1,184,731), MET?
set -u
RPT="$1"
CYC=1184731
if [ ! -f "$RPT" ]; then echo "MISSING: $RPT"; exit 1; fi

# Clock period (from Clock Summary table: "clk  {0.000 3.500}  7.000  142.857")
PERIOD=$(grep -A4 "Clock Summary" "$RPT" | grep -E "^clk " | awk '{print $(NF-1)}' | head -1)
[ -z "$PERIOD" ] && PERIOD=$(grep -E "create_clock|Requirement:" "$RPT" | grep -oE "[0-9]+\.[0-9]+ns" | head -1 | tr -d 'ns')

# WNS/WHS from Design Timing Summary: the first data row (starts with a signed
# float, has >=8 numeric-ish columns) within ~12 lines after the section header.
SUMLINE=$(awk '/Design Timing Summary/{f=1} f && /^[[:space:]]*-?[0-9]+\.[0-9]+[[:space:]]/ {print; exit}' "$RPT")
WNS=$(echo "$SUMLINE" | awk '{print $1}')
WHS=$(echo "$SUMLINE" | awk '{print $5}')

echo "RPT:        $RPT"
echo "Clock(ns):  $PERIOD"
echo "WNS(ns):    $WNS"
echo "WHS(ns):    $WHS"

# worst-path = period - WNS ; Fmax = 1000/worst-path ; fps = Fmax_Hz / CYC
awk -v p="$PERIOD" -v wns="$WNS" -v whs="$WHS" -v cyc="$CYC" 'BEGIN{
  if (p=="" || wns==""){ print "PARSE-FAIL"; exit 2 }
  wp = p - wns;
  fmax = 1000.0/wp;
  fps = (fmax*1e6)/cyc;
  printf "WorstPath:  %.3f ns\n", wp;
  printf "Fmax:       %.2f MHz\n", fmax;
  printf "fps:        %.2f  (= Fmax/%d)\n", fps, cyc;
  printf "HoldMet:    %s (WHS=%s)\n", (whs+0>=0?"YES":"NO"), whs;
  printf "MET@period: %s (constraints met iff period >= worstpath; here period=%.3f)\n", (p+0>=wp?"YES":"NO (WNS<0)"), p;
}'
