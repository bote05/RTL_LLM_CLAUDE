// Verilator testbench for rtl_library/line_buf_window.v.
//
// Tiny-geometry directed test: KH=KW=3, PH=PW=1, IC=4, IH=IW=8.
//
// What this verifies:
//
//   1. After frame_start, the entire window_flat is zero.
//
//   2. Driving the canonical scheduler walk (in_row in [0, IH+PH),
//      in_col in [0, IW+PW)) with a deterministic pixel pattern
//      produces the correct receptive-field contents at each output
//      coordinate. Compares all KH*KW*IC bytes of window_flat
//      against a Python-equivalent reference at every cycle where
//      sched_output_fires fires.
//
//   3. After the first frame completes, asserting frame_start a
//      SECOND time and rerunning the walk reproduces the exact same
//      window contents at the same cycle offsets -- no stale
//      frame-1 data leaks through. This is the multi-frame
//      regression check; the legacy module had a class of bugs
//      where line_buf cross-frame staleness corrupted the first
//      output rows of frame 2.
//
// The TB drives sched_in_row/col directly (no coord_scheduler
// instance) so the module is exercised in isolation. Pixel values
// are 1 + (row*IW + col)*IC + ic so each byte is unique within
// a frame (and the receptive field can be checked byte-by-byte).
//
// Design under test parameters:
//   IC = 4, IH = 8, IW = 8, KH = 3, KW = 3, PH = 1, PW = 1
//
//   sched_in_row width: $clog2(IH + PH + 1) = $clog2(10) = 4 bits
//   sched_in_col width: $clog2(IW + PW + 1) = $clog2(10) = 4 bits
//   data_in width:      IC*8 = 32 bits
//   window_flat width:  KH*KW*IC*8 = 288 bits
//
// Build (mirrors run_verilator's pattern; see mcp/tools.ts):
//   verilator --cc --exe --build --top line_buf_window \
//     -GIC=4 -GIH=8 -GIW=8 -GKH=3 -GKW=3 -GPH=1 -GPW=1 \
//     -I../ ../line_buf_window.v line_buf_window_tb.cpp -Wno-fatal

#include "Vline_buf_window.h"
#include <verilated.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Geometry must match the parameter overrides on the verilator command.
static constexpr int IC = 3;
static constexpr int IH = 16;
static constexpr int IW = 16;
static constexpr int KH = 7;
static constexpr int KW = 7;
static constexpr int PH = 3;
static constexpr int PW = 3;
static constexpr int OH = IH; // stride 1 in this TB (line_buf_window is stride-agnostic)
static constexpr int OW = IW;
static constexpr int MAX_IN_COL = IW + PW - 1;

// Verilator's threaded runtime references sc_time_stamp at link time.
// Provide a stub since this TB doesn't model time.
double sc_time_stamp() { return 0; }

static void tick(Vline_buf_window* dut) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

// Pixel pattern for input at row R, column C, channel ic. Each byte is
// unique within a frame so a mismatch in the receptive field is easy to
// localize. Range is 1..255 (cap at 250 to leave headroom).
static uint8_t pixel_value(int frame, int row, int col, int ic) {
    int v = ((frame & 0x7) * 64) + ((row * IW + col) * IC + ic) % 200 + 1;
    if (v > 255) v = 255;
    return static_cast<uint8_t>(v);
}

// Pack IC bytes into the data_in 32-bit word with the same layout the
// module uses: data_in[ic*8 +: 8] = pixel(row, col, ic).
static uint32_t pack_data_in(int frame, int row, int col) {
    uint32_t w = 0;
    for (int ic = 0; ic < IC; ic++) {
        w |= (static_cast<uint32_t>(pixel_value(frame, row, col, ic)) << (ic * 8));
    }
    return w;
}

// Compute the expected window contents for output coordinate (out_row,
// out_col). The receptive field is input rows (out_row - PH) ..
// (out_row - PH + KH - 1) cross input cols (out_col - PW) ..
// (out_col - PW + KW - 1). Out-of-range coords are zero (zero-padding).
//
// Returns a flat KH*KW*IC array indexed as window_flat is:
//   ref[(kh*KW*IC + kw*IC + ic)] = expected byte
static std::array<uint8_t, KH * KW * IC>
expected_window(int frame, int out_row, int out_col) {
    std::array<uint8_t, KH * KW * IC> w{};
    w.fill(0);
    for (int kh = 0; kh < KH; kh++) {
        for (int kw = 0; kw < KW; kw++) {
            int r = out_row + kh - PH;
            int c = out_col + kw - PW;
            if (r < 0 || r >= IH || c < 0 || c >= IW) continue;
            for (int ic = 0; ic < IC; ic++) {
                w[(kh * KW * IC + kw * IC + ic)] =
                    pixel_value(frame, r, c, ic);
            }
        }
    }
    return w;
}

// Read window_flat (288 bits = 36 bytes) out of the verilated DUT.
// Verilator stores wide signals as VlWide<N>, which exposes data()
// giving a uint32_t*. The signal is packed little-endian: word 0 holds
// bits [31:0], word 1 holds bits [63:32], etc.
static std::array<uint8_t, KH * KW * IC>
read_window_flat(Vline_buf_window* dut) {
    std::array<uint8_t, KH * KW * IC> w{};
    const uint32_t* p = dut->window_flat.data();
    for (int b = 0; b < KH * KW * IC; b++) {
        int word = b / 4;
        int byte_off = b % 4;
        w[b] = static_cast<uint8_t>((p[word] >> (byte_off * 8)) & 0xFF);
    }
    return w;
}

static int compare_window(int frame, int out_row, int out_col,
                          int cycle, Vline_buf_window* dut) {
    auto got = read_window_flat(dut);
    auto exp = expected_window(frame, out_row, out_col);
    int errors = 0;
    for (int b = 0; b < KH * KW * IC; b++) {
        if (got[b] != exp[b]) {
            errors++;
            if (errors <= 8) {
                int kh = b / (KW * IC);
                int kw = (b / IC) % KW;
                int ic = b % IC;
                fprintf(stderr,
                        "  mismatch frame=%d out=(%d,%d) cyc=%d "
                        "kh=%d kw=%d ic=%d got=%u expected=%u\n",
                        frame, out_row, out_col, cycle, kh, kw, ic,
                        (unsigned)got[b], (unsigned)exp[b]);
            }
        }
    }
    return errors;
}

// Run one full frame. For each cycle, drive sched_in_row/col,
// sched_advance, sched_needs_real_input, sched_output_fires, valid_in,
// data_in. We model the scheduler loosely: walk in_row 0..IH+PH-1,
// in_col 0..IW+PW-1 in row-major order; advance every cycle (no stalls
// from MAC); fire output_fires when (in_row, in_col) is the
// "next-cell-after-firing" coord = the post-advance cell after the
// firing coord in the canonical PH=1 PW=1 KH=KW=3 schedule.
//
// In the real coord_scheduler, output_fires is the registered pulse the
// cycle AFTER the scheduler advances past a firing coord. Here we model
// it directly: when (in_row, in_col) is at (r+1, c+1) after firing for
// output (r-PH+1, c-PW+1), pulse output_fires for one cycle.
//
// Since this TB doesn't include a real coord_scheduler, simplest:
// fire output_fires deterministically at the cycle BEFORE we advance
// past the firing coord. The line_buf_window doesn't actually depend
// on output_fires for correctness of the window contents; output_fires
// only freezes the shift register. We drive output_fires=0 throughout
// and check window contents at every cycle.
static int run_frame(Vline_buf_window* dut, int frame, bool check_after_pulse) {
    int errors = 0;
    int cycle = 0;

    // Frame_start pulse for one cycle.
    dut->frame_start = 1;
    dut->sched_advance = 0;
    dut->sched_needs_real_input = 0;
    dut->sched_output_fires = 0;
    dut->valid_in = 0;
    dut->data_in = 0;
    dut->sched_in_row = 0;
    dut->sched_in_col = 0;
    tick(dut);
    cycle++;
    dut->frame_start = 0;

    // Verify window is zero immediately after frame_start.
    if (check_after_pulse) {
        auto win = read_window_flat(dut);
        for (size_t b = 0; b < win.size(); b++) {
            if (win[b] != 0) {
                fprintf(stderr,
                        "FAIL frame=%d: window byte %zu = %u after frame_start "
                        "(expected 0)\n",
                        frame, b, (unsigned)win[b]);
                errors++;
                if (errors >= 4) break;
            }
        }
    }

    // Walk the canonical scheduler coords. For PH=PW=1, KH=KW=3, SH=SW=1:
    //   - in_row from 0 to IH+PH-1 = 8
    //   - in_col from 0 to IW+PW-1 = 8
    //   - sched_needs_real_input = 1 iff in_row in [0, IH) AND in_col in [0, IW)
    //   - sched_advance = 1 every cycle
    //   - valid_in tracks needs_real_input (eager upstream)
    //   - data_in carries pixel(row, col) when needs_real_input is high
    //
    // The first firing coord (for output (0,0)) is (in_row=1, in_col=1)
    // = "post-advance coord after the receptive field is complete."
    // After driving (1,1), the line_buf has rows 0 and 1 (partial), and
    // window[i][KW-1] is loaded with the rightmost column of each row.
    // We check the window at the cycle AFTER (1,1) is driven, when the
    // scheduler would assert sched_output_fires.

    for (int r = 0; r < IH + PH; r++) {
        for (int c = 0; c < IW + PW; c++) {
            bool needs_real = (r < IH) && (c < IW);
            uint32_t din = needs_real ? pack_data_in(frame, r, c) : 0;

            dut->sched_in_row = r;
            dut->sched_in_col = c;
            dut->sched_needs_real_input = needs_real ? 1 : 0;
            dut->sched_advance = 1;
            dut->sched_output_fires = 0;
            dut->valid_in = needs_real ? 1 : 0;
            dut->data_in = din;
            tick(dut);
            cycle++;

            // Check window for any output that just fired. The first
            // output (0,0) fires after driving in_row=1, in_col=1
            // (post-advance coord = receptive field for out (0,0)
            // is fully present in the window).
            //
            // At cycle of post-(r,c)-advance, the window holds the RF
            // for output coord (out_r, out_c) = (r - (KH-1-PH), c -
            // (KW-1-PW)) = (r - 1, c - 1). Valid range:
            // out_r in [0, OH), out_c in [0, OW).

            int out_r = r - (KH - 1 - PH);
            int out_c = c - (KW - 1 - PW);
            if (out_r >= 0 && out_r < OH && out_c >= 0 && out_c < OW) {
                int e = compare_window(frame, out_r, out_c, cycle, dut);
                if (e > 0) {
                    errors += e;
                    if (errors >= 32) {
                        fprintf(stderr,
                                "FAIL frame=%d: too many errors, stopping early\n",
                                frame);
                        return errors;
                    }
                }
            }
        }
    }

    // Idle for a few cycles after the frame ends.
    dut->sched_advance = 0;
    dut->sched_needs_real_input = 0;
    dut->valid_in = 0;
    dut->data_in = 0;
    for (int i = 0; i < 4; i++) tick(dut);

    return errors;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vline_buf_window* dut = new Vline_buf_window;

    // Reset.
    dut->clk = 0;
    dut->rst_n = 0;
    dut->frame_start = 0;
    dut->sched_advance = 0;
    dut->sched_needs_real_input = 0;
    dut->sched_output_fires = 0;
    dut->valid_in = 0;
    dut->data_in = 0;
    dut->sched_in_row = 0;
    dut->sched_in_col = 0;
    for (int i = 0; i < 4; i++) tick(dut);
    dut->rst_n = 1;
    tick(dut);

    int total_errors = 0;

    fprintf(stdout, "[tb] frame 0: drive canonical walk + verify window contents...\n");
    int e0 = run_frame(dut, 0, /*check_after_pulse=*/true);
    if (e0 == 0) fprintf(stdout, "[tb]   frame 0 PASS\n");
    else         fprintf(stdout, "[tb]   frame 0 FAIL (%d errors)\n", e0);
    total_errors += e0;

    fprintf(stdout, "[tb] frame 1: re-frame_start + reverify (multi-frame leak check)...\n");
    int e1 = run_frame(dut, 1, /*check_after_pulse=*/true);
    if (e1 == 0) fprintf(stdout, "[tb]   frame 1 PASS\n");
    else         fprintf(stdout, "[tb]   frame 1 FAIL (%d errors)\n", e1);
    total_errors += e1;

    fprintf(stdout, "[tb] frame 2: another reframe to stress oldest_slot wraparound...\n");
    int e2 = run_frame(dut, 2, /*check_after_pulse=*/true);
    if (e2 == 0) fprintf(stdout, "[tb]   frame 2 PASS\n");
    else         fprintf(stdout, "[tb]   frame 2 FAIL (%d errors)\n", e2);
    total_errors += e2;

    if (total_errors == 0) {
        fprintf(stdout, "[tb] ALL PASS (3 frames, %d output coords each)\n",
                OH * OW);
        delete dut;
        return 0;
    } else {
        fprintf(stdout, "[tb] FAIL: %d total mismatches\n", total_errors);
        delete dut;
        return 1;
    }
}
