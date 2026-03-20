// BETSE C++ Port - Ion Channel Models
// Complete reimplementation of all voltage-gated ion channels from Python BETSE
// channels/: vg_na.py, vg_k.py, vg_ca.py, vg_funny.py, vg_cl.py,
//            vg_morrislecar.py, wound_channel.py, cation.py, channelsabc.py
#pragma once

#include "betse_types.h"
#include <cmath>
#include <vector>
#include <string>
#include <memory>

namespace betse {

// ============================================================================
// Abstract Channel base: Hodgkin-Huxley style with m and h gates
// Mirrors channelsabc.py ChannelsABC
// ============================================================================
struct ChannelState {
    std::vector<double> m;        // activation gate
    std::vector<double> h;        // inactivation gate
    double mInf = 0, mTau = 1;   // steady-state, time constant (scalar or per-element)
    double hInf = 0, hTau = 1;
    std::vector<double> mInf_v;   // vectorised versions (when V-dependent)
    std::vector<double> mTau_v;
    std::vector<double> hInf_v;
    std::vector<double> hTau_v;
    std::vector<double> P;        // open probability per membrane
    int mpower = 1;
    int hpower = 0;
    double vrev = 0;              // reversal voltage [mV]
    double time_unit = 1.0e3;     // time unit conversion (ms for most models)
    double modulator = 1.0;       // external modulation factor

    // Ions this channel is permeable to and relative permeabilities
    std::vector<std::string> ions;
    std::vector<double> rel_perm;

    // Morris-Lecar fields
    bool kinetic_gate = true;
    double Phi = 1.0;

    // Wound channel field
    double W_factor = 2.0;
    double W_decay = 0.0;

    bool is_vectorised = false;   // whether mInf etc. are per-element
};

// ============================================================================
// Semi-implicit Euler update for m and h gates
// From channelsabc.py update_mh()
// ============================================================================
inline void update_mh(ChannelState& cs, double dt) {
    double dt_scaled = dt * cs.time_unit;
    int n = (int)cs.m.size();

    if (cs.is_vectorised) {
        for (int i = 0; i < n; i++) {
            cs.m[i] = (cs.mTau_v[i] * cs.m[i] + dt_scaled * cs.mInf_v[i]) /
                       (cs.mTau_v[i] + dt_scaled);
            cs.h[i] = (cs.hTau_v[i] * cs.h[i] + dt_scaled * cs.hInf_v[i]) /
                       (cs.hTau_v[i] + dt_scaled);
        }
    } else {
        for (int i = 0; i < n; i++) {
            cs.m[i] = (cs.mTau * cs.m[i] + dt_scaled * cs.mInf) /
                       (cs.mTau + dt_scaled);
            cs.h[i] = (cs.hTau * cs.h[i] + dt_scaled * cs.hInf) /
                       (cs.hTau + dt_scaled);
        }
    }
}

// Morris-Lecar update: from channelsabc.py update_ml()
inline void update_ml(ChannelState& cs, double dt) {
    double dt_scaled = dt * cs.time_unit;
    int n = (int)cs.m.size();
    if (cs.is_vectorised) {
        for (int i = 0; i < n; i++) {
            cs.m[i] = (cs.m[i] + (dt_scaled * cs.Phi * cs.mInf_v[i] / cs.mTau_v[i])) /
                       (1.0 + (dt_scaled * cs.Phi / cs.mTau_v[i]));
        }
    } else {
        for (int i = 0; i < n; i++) {
            cs.m[i] = (cs.m[i] + (dt_scaled * cs.Phi * cs.mInf / cs.mTau)) /
                       (1.0 + (dt_scaled * cs.Phi / cs.mTau));
        }
    }
}

// Compute open probability P = m^mpower * h^hpower
inline void compute_open_prob(ChannelState& cs, int total_mems) {
    int n = (int)cs.m.size();
    cs.P.assign(total_mems, 0.0);
    for (int i = 0; i < n; i++) {
        double p = 1.0;
        if (cs.mpower > 0) p *= std::pow(cs.m[i], cs.mpower);
        if (cs.hpower > 0) p *= std::pow(cs.h[i], cs.hpower);
        // Targets: for now, identity mapping (i -> i)
        if (i < total_mems) cs.P[i] = p;
    }
}

// ============================================================================
// Channel initialization and state calculation functions
// Each mirrors a specific Python class from the channel modules
// ============================================================================

// Helper: safe division avoiding div-by-zero in alpha/beta rate functions
inline double safe_div(double num, double den) {
    return (std::abs(den) < 1e-30) ? num / 1e-30 : num / den;
}

// ============================================================================
// VOLTAGE-GATED SODIUM CHANNELS (vg_na.py)
// ============================================================================

// --- Nav1.2 (Hammil et al 1991) ---
inline void init_Nav1p2(ChannelState& cs, const std::vector<double>& Vm_mV) {
    int n = (int)Vm_mV.size();
    cs.time_unit = 1.0e3;
    cs.vrev = 50.0;
    cs.mpower = 3;
    cs.hpower = 1;
    cs.ions = {"Na"};
    cs.rel_perm = {1.0};
    cs.m.resize(n);
    cs.h.resize(n);
    cs.mInf_v.resize(n);
    cs.mTau_v.resize(n);
    cs.hInf_v.resize(n);
    cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        double V = Vm_mV[i];
        double mA = safe_div(0.182 * ((V - 10.0) + 35.0), 1.0 - std::exp(-((V - 10.0) + 35.0) / 9.0));
        double mB = safe_div(0.124 * (-(V - 10.0) - 35.0), 1.0 - std::exp(-(-(V - 10.0) - 35.0) / 9.0));
        cs.m[i] = mA / (mA + mB);
        cs.h[i] = 1.0 / (1.0 + std::exp((V - -65.0 - 10.0) / 6.2));
    }
}

inline void calc_Nav1p2(ChannelState& cs, const std::vector<double>& Vm_mV) {
    int n = (int)Vm_mV.size();
    for (int i = 0; i < n; i++) {
        double V = Vm_mV[i];
        double mA = safe_div(0.182 * ((V - 10.0) + 35.0), 1.0 - std::exp(-((V - 10.0) + 35.0) / 9.0));
        double mB = safe_div(0.124 * (-(V - 10.0) - 35.0), 1.0 - std::exp(-(-(V - 10.0) - 35.0) / 9.0));
        cs.mInf_v[i] = mA / (mA + mB);
        cs.mTau_v[i] = 1.0 / (mA + mB);
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((V - -65.0 - 10.0) / 6.2));
        double hA_num = 0.024 * ((V - 10.0) + 50.0);
        double hA_den = 1.0 - std::exp(-((V - 10.0) + 50.0) / 5.0);
        double hB_num = 0.0091 * (-(V - 10.0) - 75.000123);
        double hB_den = 1.0 - std::exp(-(-(V - 10.0) - 75.000123) / 5.0);
        double rate = safe_div(hA_num, hA_den) + safe_div(hB_num, hB_den);
        cs.hTau_v[i] = (std::abs(rate) > 1e-30) ? 1.0 / rate : 1.0;
    }
}

// --- Nav1.3 (Cummins et al 2001) ---
inline void init_Nav1p3(ChannelState& cs, const std::vector<double>& Vm_mV) {
    int n = (int)Vm_mV.size();
    cs.time_unit = 1.0e3; cs.vrev = 50.0; cs.mpower = 3; cs.hpower = 1;
    cs.ions = {"Na"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        double V = Vm_mV[i];
        double mA = safe_div(0.182 * (V + 26.0), 1.0 - std::exp(-(V + 26.0) / 9.0));
        double mB = safe_div(0.124 * (-V - 26.0), 1.0 - std::exp(-(-V - 26.0) / 9.0));
        cs.m[i] = mA / (mA + mB);
        cs.h[i] = 1.0 / (1.0 + std::exp((V + 65.0) / 8.1));
    }
}

inline void calc_Nav1p3(ChannelState& cs, const std::vector<double>& Vm_mV) {
    int n = (int)Vm_mV.size();
    for (int i = 0; i < n; i++) {
        double V = Vm_mV[i];
        double mA = safe_div(0.182 * (V + 26.0), 1.0 - std::exp(-(V + 26.0) / 9.0));
        double mB = safe_div(0.124 * (-V - 26.0), 1.0 - std::exp(-(-V - 26.0) / 9.0));
        cs.mInf_v[i] = mA / (mA + mB);
        cs.mTau_v[i] = 1.0 / (mA + mB);
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((V + 65.0) / 8.1));
        cs.hTau_v[i] = 0.40 + 0.265 * std::exp(-V / 9.47);
    }
}

// --- NavRat1 (Huguenard 1988) ---
inline void init_NavRat1(ChannelState& cs, const std::vector<double>& Vm_mV) {
    int n = (int)Vm_mV.size();
    cs.time_unit = 1.0e3; cs.vrev = 50.0; cs.mpower = 3; cs.hpower = 1;
    cs.ions = {"Na"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        double V = Vm_mV[i];
        double mA = safe_div(0.182 * (V + 35.0), 1.0 - std::exp(-(V + 35.0) / 9.0));
        double mB = safe_div(0.124 * (-V - 35.0), 1.0 - std::exp(-(-V - 35.0) / 9.0));
        cs.m[i] = mA / (mA + mB);
        cs.h[i] = 1.0 / (1.0 + std::exp((V + 65.0) / 6.2));
    }
}

inline void calc_NavRat1(ChannelState& cs, const std::vector<double>& Vm_mV) {
    int n = (int)Vm_mV.size();
    for (int i = 0; i < n; i++) {
        double V = Vm_mV[i];
        double mA = safe_div(0.182 * (V + 35.0), 1.0 - std::exp(-(V + 35.0) / 9.0));
        double mB = safe_div(0.124 * (-V - 35.0), 1.0 - std::exp(-(-V - 35.0) / 9.0));
        cs.mInf_v[i] = mA / (mA + mB);
        cs.mTau_v[i] = 1.0 / (mA + mB);
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((V + 65.0) / 6.2));
        double hrate = safe_div(0.024 * (V + 50.0), 1.0 - std::exp(-(V + 50.0) / 5.0)) +
                       safe_div(0.0091 * (-V - 75.000123), 1.0 - std::exp(-(-V - 75.000123) / 5.0));
        cs.hTau_v[i] = (std::abs(hrate) > 1e-30) ? 1.0 / hrate : 1.0;
    }
}

// --- NavRat2 (McCormick 1992) ---
inline void init_NavRat2(ChannelState& cs, const std::vector<double>& Vm_mV) {
    int n = (int)Vm_mV.size();
    cs.time_unit = 1.0e3; cs.vrev = 50.0; cs.mpower = 3; cs.hpower = 1;
    cs.ions = {"Na"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        double V = Vm_mV[i];
        double mA = safe_div(0.091 * (V + 38.0), 1.0 - std::exp((-V - 38.0) / 5.0));
        double mB = safe_div(-0.062 * (V + 38.0), 1.0 - std::exp((V + 38.0) / 5.0));
        cs.m[i] = mA / (mA + mB);
        double hA = 0.016 * std::exp((-55.0 - V) / 15.0);
        double hB = 2.07 / (std::exp((17.0 - V) / 21.0) + 1.0);
        cs.h[i] = hA / (hA + hB);
    }
}

inline void calc_NavRat2(ChannelState& cs, const std::vector<double>& Vm_mV) {
    int n = (int)Vm_mV.size();
    for (int i = 0; i < n; i++) {
        double V = Vm_mV[i];
        double mA = safe_div(0.091 * (V + 38.0), 1.0 - std::exp((-V - 38.0) / 5.0));
        double mB = safe_div(-0.062 * (V + 38.0), 1.0 - std::exp((V + 38.0) / 5.0));
        cs.mInf_v[i] = mA / (mA + mB);
        cs.mTau_v[i] = 1.0 / (mA + mB);
        double hA = 0.016 * std::exp((-55.0 - V) / 15.0);
        double hB = 2.07 / (std::exp((17.0 - V) / 21.0) + 1.0);
        cs.hInf_v[i] = hA / (hA + hB);
        cs.hTau_v[i] = 1.0 / (hA + hB);
    }
}

// --- NavRat3 (McCormick 1992, variant) --- same kinetics as NavRat1
inline void init_NavRat3(ChannelState& cs, const std::vector<double>& Vm_mV) {
    init_NavRat1(cs, Vm_mV); // Identical kinetics
}
inline void calc_NavRat3(ChannelState& cs, const std::vector<double>& Vm_mV) {
    calc_NavRat1(cs, Vm_mV);
}

// --- Nav1.6 (Smith 1998, persistent) ---
inline void init_Nav1p6(ChannelState& cs, const std::vector<double>& Vm_mV) {
    int n = (int)Vm_mV.size();
    cs.time_unit = 1.0e3; cs.vrev = 50.0; cs.mpower = 1; cs.hpower = 0;
    cs.ions = {"Na"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n, 1.0);
    cs.is_vectorised = false;
    for (int i = 0; i < n; i++) {
        double V = Vm_mV[i];
        cs.m[i] = 1.0 / (1.0 + std::exp(-0.03937 * 4.2 * (V + 17.0)));
    }
}

inline void calc_Nav1p6(ChannelState& cs, const std::vector<double>& Vm_mV) {
    int n = (int)Vm_mV.size();
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        double V = Vm_mV[i];
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp(-0.03937 * 4.2 * (V + 17.0)));
        cs.mTau_v[i] = 1.0;
        cs.hInf_v[i] = 1.0;
        cs.hTau_v[i] = 1.0;
    }
}

// --- NaLeak (always open) ---
inline void init_NaLeak(ChannelState& cs, const std::vector<double>& Vm_mV) {
    int n = (int)Vm_mV.size();
    cs.time_unit = 1.0; cs.vrev = 50.0; cs.mpower = 0; cs.hpower = 0;
    cs.ions = {"Na"}; cs.rel_perm = {1.0};
    cs.m.assign(n, 1.0); cs.h.assign(n, 1.0);
    cs.is_vectorised = false;
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
}

inline void calc_NaLeak(ChannelState& cs, const std::vector<double>&) {
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
    cs.is_vectorised = false;
}

// ============================================================================
// VOLTAGE-GATED POTASSIUM CHANNELS (vg_k.py)
// ============================================================================

// Helper for Boltzmann-type K channels
struct BoltzmannParams {
    double mV_half, m_slope, mTau_max, mTau_half, mTau_slope;
    double hV_half, h_slope, hTau_max, hTau_half, hTau_slope;
    int mpower, hpower;
    double vrev;
};

inline void init_boltzmann_K(ChannelState& cs, const std::vector<double>& Vm_mV,
                              const BoltzmannParams& bp) {
    int n = (int)Vm_mV.size();
    cs.time_unit = 1.0e3; cs.vrev = bp.vrev; cs.mpower = bp.mpower; cs.hpower = bp.hpower;
    cs.ions = {"K"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.is_vectorised = false;
    for (int i = 0; i < n; i++) {
        double V = Vm_mV[i];
        cs.m[i] = 1.0 / (1.0 + std::exp((V - bp.mV_half) / bp.m_slope));
        cs.h[i] = (bp.hpower > 0) ? 1.0 / (1.0 + std::exp((V - bp.hV_half) / bp.h_slope)) : 1.0;
    }
}

inline void calc_boltzmann_K(ChannelState& cs, const std::vector<double>& Vm_mV,
                              const BoltzmannParams& bp) {
    int n = (int)Vm_mV.size();
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        double V = Vm_mV[i];
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V - bp.mV_half) / bp.m_slope));
        cs.mTau_v[i] = bp.mTau_max / (1.0 + std::exp((V - bp.mTau_half) / bp.mTau_slope));
        cs.hInf_v[i] = (bp.hpower > 0) ? 1.0 / (1.0 + std::exp((V - bp.hV_half) / bp.h_slope)) : 1.0;
        cs.hTau_v[i] = (bp.hpower > 0) ? bp.hTau_max / (1.0 + std::exp((V - bp.hTau_half) / bp.hTau_slope)) : 1.0;
    }
}

// Kv1.1
inline void init_Kv1p1(ChannelState& cs, const std::vector<double>& V) {
    init_boltzmann_K(cs, V, {-30.5, -11.3943, 30.0, -76.56, 26.1479, -30.0, 27.3943, 15000.0, -160.56, -100.0, 1, 2, -65.0});
}
inline void calc_Kv1p1(ChannelState& cs, const std::vector<double>& V) {
    calc_boltzmann_K(cs, V, {-30.5, -11.3943, 30.0, -76.56, 26.1479, -30.0, 27.3943, 15000.0, -160.56, -100.0, 1, 2, -65.0});
}

// Kv1.2
inline void init_Kv1p2(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = -65; cs.mpower = 1; cs.hpower = 1;
    cs.ions = {"K"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.m[i] = 1.0 / (1.0 + std::exp(-(V[i] + 21.0) / 11.3943));
        cs.h[i] = 1.0 / (1.0 + std::exp((V[i] + 22.0) / 11.3943));
    }
}
inline void calc_Kv1p2(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp(-(V[i] + 21.0) / 11.3943));
        cs.mTau_v[i] = 150.0 / (1.0 + std::exp((V[i] + 67.56) / 34.1479));
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 22.0) / 11.3943));
        cs.hTau_v[i] = 15000.0 / (1.0 + std::exp(-(V[i] + 46.56) / 44.1479));
    }
}

// Kv1.3
inline void init_Kv1p3(ChannelState& cs, const std::vector<double>& V) {
    init_boltzmann_K(cs, V, {-14.1, -10.3, 1.0, 0, 1.0, -33.0, 3.7, 1.0, 0, 1.0, 1, 1, -65.0});
    // Override tau with linear models
}
inline void calc_Kv1p3(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 14.1) / -10.3));
        cs.mTau_v[i] = -0.284 * V[i] + 19.16;
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 33.0) / 3.7));
        cs.hTau_v[i] = -13.76 * V[i] + 1162.4;
    }
}

// Kv1.4
inline void init_Kv1p4(ChannelState& cs, const std::vector<double>& V) {
    init_boltzmann_K(cs, V, {-21.7, -16.9, 3.0, 0, 1.0, -73.6, 12.8, 119.0, 0, 1.0, 1, 1, -65.0});
}
inline void calc_Kv1p4(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 21.7) / -16.9));
        cs.mTau_v[i] = 3.0;
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 73.6) / 12.8));
        cs.hTau_v[i] = 119.0;
    }
}

// Kv1.5
inline void init_Kv1p5(ChannelState& cs, const std::vector<double>& V) {
    init_boltzmann_K(cs, V, {-6.0, -6.4, 1.0, 0, 1.0, -25.3, 3.5, 1.0, 0, 1.0, 1, 1, -65.0});
}
inline void calc_Kv1p5(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 6.0) / -6.4));
        cs.mTau_v[i] = -0.1163 * V[i] + 8.33;
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 25.3) / 3.5));
        cs.hTau_v[i] = -15.5 * V[i] + 1620.0;
    }
}

// Kv1.6
inline void init_Kv1p6(ChannelState& cs, const std::vector<double>& V) {
    init_boltzmann_K(cs, V, {-20.8, -8.1, 30.0, -46.56, 44.14, -22.0, 11.39, 5000.0, -46.56, -44.14, 1, 1, -65.0});
}
inline void calc_Kv1p6(ChannelState& cs, const std::vector<double>& V) {
    calc_boltzmann_K(cs, V, {-20.8, -8.1, 30.0, -46.56, 44.14, -22.0, 11.39, 5000.0, -46.56, -44.14, 1, 1, -65.0});
}

// Kv2.1
inline void init_Kv2p1(ChannelState& cs, const std::vector<double>& V) {
    init_boltzmann_K(cs, V, {-9.2, -6.6, 100.0, -46.56, 44.14, -19.0, 5.0, 10000.0, -46.56, -44.14, 1, 1, -65.0});
}
inline void calc_Kv2p1(ChannelState& cs, const std::vector<double>& V) {
    calc_boltzmann_K(cs, V, {-9.2, -6.6, 100.0, -46.56, 44.14, -19.0, 5.0, 10000.0, -46.56, -44.14, 1, 1, -65.0});
}

// Kv2.2
inline void init_Kv2p2(ChannelState& cs, const std::vector<double>& V) {
    init_boltzmann_K(cs, V, {5.0, -12.0, 130.0, -46.56, -44.14, -16.3, 4.8, 10000.0, -46.56, -44.14, 1, 1, -65.0});
}
inline void calc_Kv2p2(ChannelState& cs, const std::vector<double>& V) {
    calc_boltzmann_K(cs, V, {5.0, -12.0, 130.0, -46.56, -44.14, -16.3, 4.8, 10000.0, -46.56, -44.14, 1, 1, -65.0});
}

// Kv3.1
inline void init_Kv3p1(ChannelState& cs, const std::vector<double>& V) {
    init_boltzmann_K(cs, V, {18.7, -9.7, 20.0, -46.56, -44.14, 0, 1.0, 1.0, 0, 1.0, 1, 0, -65.0});
}
inline void calc_Kv3p1(ChannelState& cs, const std::vector<double>& V) {
    calc_boltzmann_K(cs, V, {18.7, -9.7, 20.0, -46.56, -44.14, 0, 1.0, 1.0, 0, 1.0, 1, 0, -65.0});
}

// Kv3.2
inline void init_Kv3p2(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = -65; cs.mpower = 2; cs.hpower = 0;
    cs.ions = {"K"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.assign(n, 1.0);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.m[i] = 1.0 / (1.0 + std::exp((V[i] + 0.373267) / -8.568187));
    }
}
inline void calc_Kv3p2(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 0.373267) / -8.568187));
        cs.mTau_v[i] = 3.241643 + 19.106496 / (1.0 + std::exp((V[i] - 19.220623) / 4.451533));
        cs.hInf_v[i] = 1.0;
        cs.hTau_v[i] = 1.0;
    }
}

// Kv3.3
inline void init_Kv3p3(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = 82.0; cs.mpower = 2; cs.hpower = 1;
    cs.ions = {"K"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.m[i] = 1.0 / (1.0 + std::exp((V[i] - 35.0) / -7.3));
        cs.h[i] = 0.25 + 0.75 / (1.0 + std::exp((V[i] + 28.293856) / 29.385636));
    }
}
inline void calc_Kv3p3(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] - 35.0) / -7.3));
        cs.mTau_v[i] = 0.676808 + 27.913114 / (1.0 + std::exp((V[i] - 22.414149) / 9.704638));
        cs.hInf_v[i] = 0.25 + 0.75 / (1.0 + std::exp((V[i] + 28.293856) / 29.385636));
        cs.hTau_v[i] = 199.786728 + 2776.119438 * std::exp(-V[i] / 7.309565);
    }
}

// Kv3.4
inline void init_Kv3p4(ChannelState& cs, const std::vector<double>& V) {
    init_boltzmann_K(cs, V, {-3.4, -8.4, 10.0, 4.44, 38.14, -53.32, 7.4, 20000.0, -46.56, -44.14, 1, 1, -65.0});
}
inline void calc_Kv3p4(ChannelState& cs, const std::vector<double>& V) {
    calc_boltzmann_K(cs, V, {-3.4, -8.4, 10.0, 4.44, 38.14, -53.32, 7.4, 20000.0, -46.56, -44.14, 1, 1, -65.0});
}

// K_Fast (Korngreen 2000)
inline void init_K_Fast(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = -65; cs.mpower = 1; cs.hpower = 1;
    cs.ions = {"K"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.m[i] = 1.0 / (1.0 + std::exp(-(V[i] + 47.0) / 29.0));
        cs.h[i] = 1.0 / (1.0 + std::exp((V[i] + 56.0) / 10.0));
    }
}
inline void calc_K_Fast(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp(-(V[i] + 47.0) / 29.0));
        cs.mTau_v[i] = 0.34 + 0.92 * std::exp(-std::pow((V[i] + 71.0) / 59.0, 2));
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 56.0) / 10.0));
        cs.hTau_v[i] = 8.0 + 49.0 * std::exp(-std::pow((V[i] + 73.0) / 23.0, 2));
    }
}

// KLeak (always open)
inline void init_KLeak(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0; cs.vrev = -65; cs.mpower = 0; cs.hpower = 0;
    cs.ions = {"K"}; cs.rel_perm = {1.0};
    cs.m.assign(n, 1.0); cs.h.assign(n, 1.0);
    cs.is_vectorised = false;
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
}
inline void calc_KLeak(ChannelState& cs, const std::vector<double>&) {
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
    cs.is_vectorised = false;
}

// Kir2.1 (inward rectifier)
inline void init_Kir2p1(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = -70.6; cs.mpower = 1; cs.hpower = 2;
    cs.ions = {"K"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.m[i] = 1.0 / (1.0 + std::exp((V[i] + 96.48) / 23.26));
        cs.h[i] = 1.0 / (1.0 + std::exp((V[i] + 168.28) / -44.13));
    }
}
inline void calc_Kir2p1(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 96.48) / 23.26));
        cs.mTau_v[i] = 3.7 + (-3.37 / (1.0 + std::exp((V[i] + 32.9) / 27.93)));
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 168.28) / -44.13));
        cs.hTau_v[i] = 0.85 + 306.3 / (1.0 + std::exp((V[i] + 118.29) / -27.23));
    }
}

// ============================================================================
// VOLTAGE-GATED CALCIUM CHANNELS (vg_ca.py)
// ============================================================================

// Cav3.3 (T-type, Traboulsie 2007)
inline void init_Cav3p3(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = 30; cs.mpower = 1; cs.hpower = 1;
    cs.ions = {"Ca"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.m[i] = 1.0 / (1.0 + std::exp((V[i] + 45.454426) / -5.073015));
        cs.h[i] = 1.0 / (1.0 + std::exp((V[i] + 74.031965) / 8.416382));
    }
}
inline void calc_Cav3p3(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 45.454426) / -5.073015));
        cs.mTau_v[i] = 3.394938 + 54.187616 / (1.0 + std::exp((V[i] + 40.040397) / 4.110392));
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 74.031965) / 8.416382));
        cs.hTau_v[i] = 109.701136 + 0.003816 * std::exp(-V[i] / 4.781719);
    }
}

// Cav2.1 (P/Q-type, Miyasho 2001)
inline void init_Cav2p1(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = 135; cs.mpower = 1; cs.hpower = 0;
    cs.ions = {"Ca"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.assign(n, 1.0);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        double mA = 8.5 / (1.0 + std::exp((V[i] - 8.0) / -12.5));
        double mB = 35.0 / (1.0 + std::exp((V[i] + 74.0) / 14.5));
        cs.m[i] = mA / (mA + mB);
    }
}
inline void calc_Cav2p1(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        double mA = 8.5 / (1.0 + std::exp((V[i] - 8.0) / -12.5));
        double mB = 35.0 / (1.0 + std::exp((V[i] + 74.0) / 14.5));
        cs.mInf_v[i] = mA / (mA + mB);
        cs.mTau_v[i] = 1.0 / (mA + mB);
        cs.hInf_v[i] = 1.0; cs.hTau_v[i] = 1.0;
    }
}

// Cav1.3 (L-type LVA, Avery 1996)
inline void init_Cav1p3(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = 113; cs.mpower = 2; cs.hpower = 1;
    cs.ions = {"Ca"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.m[i] = 1.0 / (1.0 + std::exp((V[i] + 30.0) / -6.0));
        cs.h[i] = 1.0 / (1.0 + std::exp((V[i] + 80.0) / 6.4));
    }
}
inline void calc_Cav1p3(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 30.0) / -6.0));
        cs.mTau_v[i] = 5.0 + 20.0 / (1.0 + std::exp((V[i] + 25.0) / 5.0));
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 80.0) / 6.4));
        cs.hTau_v[i] = 20.0 + 50.0 / (1.0 + std::exp((V[i] + 40.0) / 7.0));
    }
}

// Cav1.2 (L-type, Carlin 2000)
inline void init_Cav1p2(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = 131; cs.mpower = 2; cs.hpower = 1;
    cs.ions = {"Ca"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        double Vs = V[i] - 10.0;
        cs.m[i] = 1.0 / (1.0 + std::exp((Vs + 30.0) / -6.0));
        cs.h[i] = 1.0 / (1.0 + std::exp((Vs + 80.0) / 6.4));
    }
}
inline void calc_Cav1p2(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        double Vs = V[i] - 10.0;
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((Vs + 30.0) / -6.0));
        cs.mTau_v[i] = 5.0 + 20.0 / (1.0 + std::exp((Vs + 25.0) / 5.0));
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((Vs + 80.0) / 6.4));
        cs.hTau_v[i] = 20.0 + 50.0 / (1.0 + std::exp((Vs + 40.0) / 7.0));
    }
}

// Cav2.3 (R-type, Miyasho 2001)
inline void init_Cav2p3(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = 30; cs.mpower = 1; cs.hpower = 1;
    cs.ions = {"Ca"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        double mA = 2.6 / (1.0 + std::exp((V[i] + 7.0) / -8.0));
        double mB = 0.18 / (1.0 + std::exp((V[i] + 26.0) / 4.0));
        double hA = 0.0025 / (1.0 + std::exp((V[i] + 32.0) / 8.0));
        double hB = 0.19 / (1.0 + std::exp((V[i] + 42.0) / -10.0));
        cs.m[i] = mA / (mA + mB);
        cs.h[i] = hA / (hA + hB);
    }
}
inline void calc_Cav2p3(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        double mA = 2.6 / (1.0 + std::exp((V[i] + 7.0) / -8.0));
        double mB = 0.18 / (1.0 + std::exp((V[i] + 26.0) / 4.0));
        double hA = 0.0025 / (1.0 + std::exp((V[i] + 32.0) / 8.0));
        double hB = 0.19 / (1.0 + std::exp((V[i] + 42.0) / -10.0));
        cs.mInf_v[i] = mA / (mA + mB);
        cs.mTau_v[i] = 1.0 / (mA + mB);
        cs.hInf_v[i] = hA / (hA + hB);
        cs.hTau_v[i] = 1.0 / (hA + hB);
    }
}

// Cav2.2 (N-type, Huang 1998)
inline void init_Cav2p2(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = 135; cs.mpower = 2; cs.hpower = 1;
    cs.ions = {"Ca"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        double mA = safe_div(0.1 * (V[i] - 20.0), 1.0 - std::exp(-(V[i] - 20.0) / 10.0));
        double mB = 0.4 * std::exp(-(V[i] + 25.0) / 18.0);
        double hA = 0.01 * std::exp(-(V[i] + 50.0) / 10.0);
        double hB = 0.1 / (1.0 + std::exp(-(V[i] + 17.0) / 17.0));
        cs.m[i] = mA / (mA + mB);
        cs.h[i] = hA / (hA + hB);
    }
}
inline void calc_Cav2p2(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        double mA = safe_div(0.1 * (V[i] - 20.0), 1.0 - std::exp(-(V[i] - 20.0) / 10.0));
        double mB = 0.4 * std::exp(-(V[i] + 25.0) / 18.0);
        double hA = 0.01 * std::exp(-(V[i] + 50.0) / 10.0);
        double hB = 0.1 / (1.0 + std::exp(-(V[i] + 17.0) / 17.0));
        cs.mInf_v[i] = mA / (mA + mB);
        cs.mTau_v[i] = 1.0 / (mA + mB);
        cs.hInf_v[i] = hA / (hA + hB);
        cs.hTau_v[i] = 1.0 / (hA + hB);
    }
}

// Cav3.1 (T-type, Traboulsie 2007)
inline void init_Cav3p1(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = 30; cs.mpower = 1; cs.hpower = 1;
    cs.ions = {"Ca"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.m[i] = 1.0 / (1.0 + std::exp((V[i] + 42.921064) / -5.163208));
        cs.h[i] = 1.0 / (1.0 + std::exp((V[i] + 72.90742) / 4.575763));
    }
}
inline void calc_Cav3p1(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 42.921064) / -5.163208));
        double mt = -0.855809 + 1.493527 * std::exp(-V[i] / 27.414182);
        cs.mTau_v[i] = std::max(mt, 1.0);
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 72.90742) / 4.575763));
        cs.hTau_v[i] = 9.987873 + 0.002883 * std::exp(-V[i] / 5.598574);
    }
}

// Ca_L2, Ca_L3 (L-type variants)
inline void init_Ca_L2(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = 131; cs.mpower = 2; cs.hpower = 1;
    cs.ions = {"Ca"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.is_vectorised = false;
    for (int i = 0; i < n; i++) {
        cs.m[i] = 1.0 / (1.0 + std::exp((V[i] + 30.0) / -6.0));
        cs.h[i] = 1.0 / (1.0 + std::exp((V[i] + 80.0) / 6.4));
    }
}
inline void calc_Ca_L2(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 30.0) / -6.0));
        cs.mTau_v[i] = 10.0;
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 80.0) / 6.4));
        cs.hTau_v[i] = 59.0;
    }
}

inline void init_Ca_L3(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = 131; cs.mpower = 2; cs.hpower = 1;
    cs.ions = {"Ca"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.is_vectorised = false;
    for (int i = 0; i < n; i++) {
        double Vs = V[i] - 15.0;
        cs.m[i] = 1.0 / (1.0 + std::exp((Vs + 30.0) / -6.0));
        cs.h[i] = 1.0 / (1.0 + std::exp((Vs + 80.0) / 6.4));
    }
}
inline void calc_Ca_L3(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        double Vs = V[i] - 15.0;
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((Vs + 30.0) / -6.0));
        cs.mTau_v[i] = 10.0;
        cs.hInf_v[i] = 1.0 / (1.0 + std::exp((Vs + 80.0) / 6.4));
        cs.hTau_v[i] = 59.0;
    }
}

// Cav_G (general, Reuveni 1993)
inline void init_Cav_G(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = 131; cs.mpower = 2; cs.hpower = 1;
    cs.ions = {"Ca"}; cs.rel_perm = {1.0};
    cs.m.resize(n); cs.h.resize(n);
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        double mA = safe_div(0.055 * (-27.0 - V[i]), std::exp((-27.0 - V[i]) / 3.8) - 1.0);
        double mB = 0.94 * std::exp((-75.0 - V[i]) / 17.0);
        double hA = 0.000457 * std::exp((-13.0 - V[i]) / 50.0);
        double hB = 0.0065 / (std::exp((-V[i] - 15.0) / 28.0) + 1.0);
        cs.m[i] = mA / (mA + mB);
        cs.h[i] = hA / (hA + hB);
    }
}
inline void calc_Cav_G(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        double mA = safe_div(0.055 * (-27.0 - V[i]), std::exp((-27.0 - V[i]) / 3.8) - 1.0);
        double mB = 0.94 * std::exp((-75.0 - V[i]) / 17.0);
        double hA = 0.000457 * std::exp((-13.0 - V[i]) / 50.0);
        double hB = 0.0065 / (std::exp((-V[i] - 15.0) / 28.0) + 1.0);
        cs.mInf_v[i] = mA / (mA + mB);
        cs.mTau_v[i] = 1.0 / (mA + mB);
        cs.hInf_v[i] = hA / (hA + hB);
        cs.hTau_v[i] = 1.0 / (hA + hB);
    }
}

// CaLeak
inline void init_CaLeak(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0; cs.vrev = 30; cs.mpower = 0; cs.hpower = 0;
    cs.ions = {"Ca"}; cs.rel_perm = {1.0};
    cs.m.assign(n, 1.0); cs.h.assign(n, 1.0);
    cs.is_vectorised = false;
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
}
inline void calc_CaLeak(ChannelState& cs, const std::vector<double>&) {
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
    cs.is_vectorised = false;
}

// ============================================================================
// HCN / FUNNY CURRENT CHANNELS (vg_funny.py)
// ============================================================================

// HCN1
inline void init_HCN1(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = -45; cs.mpower = 1; cs.hpower = 0;
    cs.ions = {"Na", "K", "Ca"}; cs.rel_perm = {0.2, 1.0, 0.05};
    cs.m.resize(n); cs.h.assign(n, 1.0);
    cs.is_vectorised = false;
    for (int i = 0; i < n; i++)
        cs.m[i] = 1.0 / (1.0 + std::exp((V[i] + 94.0) / 8.1));
}
inline void calc_HCN1(ChannelState& cs, const std::vector<double>& V) {
    cs.vrev = -45;
    int n = (int)V.size();
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] + 94.0) / 8.1));
        cs.mTau_v[i] = 30.0;
        cs.hInf_v[i] = 1.0; cs.hTau_v[i] = 1.0;
    }
}

// HCN2
inline void init_HCN2(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = -45; cs.mpower = 1; cs.hpower = 0;
    cs.ions = {"Na", "K", "Ca"}; cs.rel_perm = {0.2, 1.0, 0.05};
    cs.m.resize(n); cs.h.assign(n, 1.0);
    cs.is_vectorised = false;
    for (int i = 0; i < n; i++)
        cs.m[i] = 1.0 / (1.0 + std::exp((V[i] - 10.0 + 99.0) / 6.2));
}
inline void calc_HCN2(ChannelState& cs, const std::vector<double>& V) {
    cs.vrev = -45;
    int n = (int)V.size();
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] - 10.0 + 99.0) / 6.2));
        cs.mTau_v[i] = 184.0;
        cs.hInf_v[i] = 1.0; cs.hTau_v[i] = 1.0;
    }
}

// HCN4
inline void init_HCN4(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = -45; cs.mpower = 1; cs.hpower = 0;
    cs.ions = {"Na", "K", "Ca"}; cs.rel_perm = {0.2, 1.0, 0.05};
    cs.m.resize(n); cs.h.assign(n, 1.0);
    cs.is_vectorised = false;
    for (int i = 0; i < n; i++)
        cs.m[i] = 1.0 / (1.0 + std::exp((V[i] - 10.0 + 100.0) / 9.6));
}
inline void calc_HCN4(ChannelState& cs, const std::vector<double>& V) {
    cs.vrev = -45;
    int n = (int)V.size();
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n);
    cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] - 10.0 + 100.0) / 9.6));
        cs.mTau_v[i] = 461.0;
        cs.hInf_v[i] = 1.0; cs.hTau_v[i] = 1.0;
    }
}

// HCNLeak
inline void init_HCNLeak(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0; cs.vrev = -45; cs.mpower = 0; cs.hpower = 0;
    cs.ions = {"Na", "K", "Ca"}; cs.rel_perm = {0.33, 1.0, 0.05};
    cs.m.assign(n, 1.0); cs.h.assign(n, 1.0);
    cs.is_vectorised = false;
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
}
inline void calc_HCNLeak(ChannelState& cs, const std::vector<double>&) {
    cs.vrev = -45; cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
    cs.is_vectorised = false;
}

// HCN2_cAMP and HCN4_cAMP (cAMP-shifted variants)
inline void init_HCN2_cAMP(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = -45; cs.mpower = 1; cs.hpower = 0;
    cs.ions = {"Na", "K", "Ca"}; cs.rel_perm = {0.2, 1.0, 0.05};
    cs.m.resize(n); cs.h.assign(n, 1.0); cs.is_vectorised = false;
    for (int i = 0; i < n; i++)
        cs.m[i] = 1.0 / (1.0 + std::exp((V[i] - 20.0 + 99.0) / 6.2));
}
inline void calc_HCN2_cAMP(ChannelState& cs, const std::vector<double>& V) {
    cs.vrev = -45; int n = (int)V.size();
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n); cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] - 20.0 + 99.0) / 6.2));
        cs.mTau_v[i] = 184.0; cs.hInf_v[i] = 1; cs.hTau_v[i] = 1;
    }
}

inline void init_HCN4_cAMP(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = -45; cs.mpower = 1; cs.hpower = 0;
    cs.ions = {"Na", "K", "Ca"}; cs.rel_perm = {0.2, 1.0, 0.05};
    cs.m.resize(n); cs.h.assign(n, 1.0); cs.is_vectorised = false;
    for (int i = 0; i < n; i++)
        cs.m[i] = 1.0 / (1.0 + std::exp((V[i] - 20.0 + 100.0) / 9.6));
}
inline void calc_HCN4_cAMP(ChannelState& cs, const std::vector<double>& V) {
    cs.vrev = -45; int n = (int)V.size();
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n); cs.hTau_v.resize(n); cs.is_vectorised = true;
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / (1.0 + std::exp((V[i] - 24.0 + 100.0) / 9.6));
        cs.mTau_v[i] = 461.0; cs.hInf_v[i] = 1; cs.hTau_v[i] = 1;
    }
}

// ============================================================================
// CHLORIDE CHANNELS (vg_cl.py)
// ============================================================================
inline void init_ClLeak(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0; cs.vrev = -65; cs.mpower = 0; cs.hpower = 0;
    cs.ions = {"Cl"}; cs.rel_perm = {1.0};
    cs.m.assign(n, 1.0); cs.h.assign(n, 1.0);
    cs.is_vectorised = false;
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
}
inline void calc_ClLeak(ChannelState& cs, const std::vector<double>&) {
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
    cs.is_vectorised = false;
}

// ============================================================================
// CATION LEAK CHANNELS (cation.py)
// ============================================================================
inline void init_CatLeak(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0; cs.mpower = 0; cs.hpower = 0;
    cs.ions = {"Na", "K", "Ca"}; cs.rel_perm = {1.0, 1.0, 0.0};
    cs.m.assign(n, 1.0); cs.h.assign(n, 1.0);
    cs.is_vectorised = false;
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
}
inline void calc_CatLeak(ChannelState& cs, const std::vector<double>&) {
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1; cs.is_vectorised = false;
}

inline void init_CatLeak2(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0; cs.mpower = 0; cs.hpower = 0;
    cs.ions = {"Na", "K", "Ca"}; cs.rel_perm = {1.0, 1.0, 1.0};
    cs.m.assign(n, 1.0); cs.h.assign(n, 1.0);
    cs.is_vectorised = false;
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
}
inline void calc_CatLeak2(ChannelState& cs, const std::vector<double>&) {
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1; cs.is_vectorised = false;
}

// ============================================================================
// MORRIS-LECAR SIMPLIFIED CHANNELS (vg_morrislecar.py)
// ============================================================================

// Helper for ML channels
inline void init_ML(ChannelState& cs, const std::vector<double>& V,
                     double v_half, double v_slope, double vrev,
                     const std::vector<std::string>& ions,
                     const std::vector<double>& perms,
                     bool kinetic, double phi,
                     bool use_cosh = false) {
    int n = (int)V.size();
    cs.time_unit = 1.0e3; cs.vrev = vrev; cs.mpower = 1; cs.hpower = 0;
    cs.ions = ions; cs.rel_perm = perms;
    cs.kinetic_gate = kinetic; cs.Phi = phi;
    cs.m.resize(n); cs.h.assign(n, 1.0);
    cs.is_vectorised = true;
    cs.mInf_v.resize(n); cs.mTau_v.resize(n);
    cs.hInf_v.resize(n, 1.0); cs.hTau_v.resize(n, 1.0);
    for (int i = 0; i < n; i++) {
        if (use_cosh)
            cs.m[i] = 1.0 / std::cosh((V[i] - v_half) / v_slope);
        else
            cs.m[i] = 0.5 * (1.0 + std::tanh((V[i] - v_half) / v_slope));
    }
}
inline void calc_ML(ChannelState& cs, const std::vector<double>& V,
                     double v_half, double v_slope, bool use_cosh = false) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        if (use_cosh) {
            cs.mInf_v[i] = 1.0 / std::cosh((V[i] - v_half) / v_slope);
        } else {
            cs.mInf_v[i] = 0.5 * (1.0 + std::tanh((V[i] - v_half) / v_slope));
        }
        cs.mTau_v[i] = 1.0 / std::cosh((V[i] - v_half) / (2.0 * v_slope));
    }
}

// Concrete ML channels
inline void init_Kv_ML1(ChannelState& cs, const std::vector<double>& V) {
    init_ML(cs, V, 12.0, 17.0, -85.0, {"K"}, {1.0}, true, 0.066);
}
inline void calc_Kv_ML1(ChannelState& cs, const std::vector<double>& V) {
    calc_ML(cs, V, 12.0, 17.0);
}

inline void init_Kv2p1_ML(ChannelState& cs, const std::vector<double>& V) {
    init_ML(cs, V, 14.0, 30.0, -65.0, {"K"}, {1.0}, true, 0.066);
}
inline void calc_Kv2p1_ML(ChannelState& cs, const std::vector<double>& V) {
    calc_ML(cs, V, 14.0, 30.0);
}

inline void init_Kv1p3_ML(ChannelState& cs, const std::vector<double>& V) {
    init_ML(cs, V, -14.0, 20.0, -65.0, {"K"}, {1.0}, true, 0.066);
}
inline void calc_Kv1p3_ML(ChannelState& cs, const std::vector<double>& V) {
    calc_ML(cs, V, -14.0, 20.0);
}

inline void init_Kv1p5_ML(ChannelState& cs, const std::vector<double>& V) {
    init_ML(cs, V, -6.0, 15.0, -65.0, {"K"}, {1.0}, true, 0.066);
}
inline void calc_Kv1p5_ML(ChannelState& cs, const std::vector<double>& V) {
    calc_ML(cs, V, -6.0, 15.0);
}

inline void init_Kv1p5S_ML(ChannelState& cs, const std::vector<double>& V) {
    init_ML(cs, V, -6.0, 15.0, -65.0, {"K"}, {1.0}, true, 0.00132);
}
inline void calc_Kv1p5S_ML(ChannelState& cs, const std::vector<double>& V) {
    calc_ML(cs, V, -6.0, 15.0);
}

inline void init_Nav_ML(ChannelState& cs, const std::vector<double>& V) {
    init_ML(cs, V, -17.0, 18.0, 50.0, {"Na"}, {1.0}, false, 1.0);
}
inline void calc_Nav_ML(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 0.5 * (1.0 + std::tanh((V[i] + 17.0) / 18.0));
        cs.mTau_v[i] = 1.0;
    }
}

inline void init_Cav_L_ML(ChannelState& cs, const std::vector<double>& V) {
    init_ML(cs, V, -20.0, 24.0, 80.0, {"Ca"}, {1.0}, false, 1.0);
}
inline void calc_Cav_L_ML(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 0.5 * (1.0 + std::tanh((V[i] + 20.0) / 24.0));
        cs.mTau_v[i] = 1.0;
    }
}

inline void init_Cav_L_ML2(ChannelState& cs, const std::vector<double>& V) {
    init_ML(cs, V, -20.0, 12.0, 80.0, {"Ca"}, {1.0}, false, 1.0);
}
inline void calc_Cav_L_ML2(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 0.5 * (1.0 + std::tanh((V[i] + 20.0) / 12.0));
        cs.mTau_v[i] = 1.0;
    }
}

inline void init_Cav_N_ML(ChannelState& cs, const std::vector<double>& V) {
    init_ML(cs, V, -1.0, 18.0, 80.0, {"Ca"}, {1.0}, false, 1.0);
}
inline void calc_Cav_N_ML(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 0.5 * (1.0 + std::tanh((V[i] + 1.0) / 18.0));
        cs.mTau_v[i] = 1.0;
    }
}

inline void init_Cav_T_ML(ChannelState& cs, const std::vector<double>& V) {
    init_ML(cs, V, -43.0, 24.0, 80.0, {"Ca"}, {1.0}, false, 1.0);
}
inline void calc_Cav_T_ML(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 0.5 * (1.0 + std::tanh((V[i] + 43.0) / 24.0));
        cs.mTau_v[i] = 1.0;
    }
}

inline void init_Kir_ML(ChannelState& cs, const std::vector<double>& V) {
    init_ML(cs, V, -135.0, 37.0, -65.0, {"K"}, {1.0}, false, 1.0, true);
}
inline void calc_Kir_ML(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 1.0 / std::cosh((V[i] + 135.0) / 37.0);
        cs.mTau_v[i] = 1.0;
    }
}

inline void init_HCN2_ML(ChannelState& cs, const std::vector<double>& V) {
    init_ML(cs, V, -99.0, 12.4, -65.0, {"K", "Na"}, {1.0, 0.2}, false, 1.0);
}
inline void calc_HCN2_ML(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 0.5 * (1.0 + std::tanh((V[i] + 99.0) / 12.4));
        cs.mTau_v[i] = 1.0;
    }
}

inline void init_HCN4_ML(ChannelState& cs, const std::vector<double>& V) {
    init_ML(cs, V, -99.0, 19.2, -65.0, {"K", "Na"}, {1.0, 0.2}, false, 1.0);
}
inline void calc_HCN4_ML(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    for (int i = 0; i < n; i++) {
        cs.mInf_v[i] = 0.5 * (1.0 + std::tanh((V[i] + 99.0) / 19.2));
        cs.mTau_v[i] = 1.0;
    }
}

// ============================================================================
// WOUND CHANNEL (wound_channel.py)
// ============================================================================
inline void init_TRP(ChannelState& cs, const std::vector<double>& V) {
    int n = (int)V.size();
    cs.time_unit = 1.0; cs.vrev = 0; cs.mpower = 0; cs.hpower = 0;
    cs.ions = {"Na", "K"}; cs.rel_perm = {1.0, 1.0};
    cs.m.assign(n, 1.0); cs.h.assign(n, 1.0);
    cs.W_factor = 2.0;
    cs.is_vectorised = false;
    cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
}
inline void calc_TRP(ChannelState& cs, const std::vector<double>&) {
    cs.vrev = 0; cs.mInf = 1; cs.mTau = 1; cs.hInf = 1; cs.hTau = 1;
    cs.is_vectorised = false;
}

// ============================================================================
// Channel dispatch: initialize and calculate by ChannelType enum
// ============================================================================
inline void channel_init(ChannelState& cs, ChannelType type, const std::vector<double>& Vm_mV) {
    switch (type) {
        case ChannelType::Nav1p2: init_Nav1p2(cs, Vm_mV); break;
        case ChannelType::Nav1p3: init_Nav1p3(cs, Vm_mV); break;
        case ChannelType::Nav1p6: init_Nav1p6(cs, Vm_mV); break;
        case ChannelType::NavRat1: init_NavRat1(cs, Vm_mV); break;
        case ChannelType::NavRat2: init_NavRat2(cs, Vm_mV); break;
        case ChannelType::NavRat3: init_NavRat3(cs, Vm_mV); break;
        case ChannelType::NaLeak: init_NaLeak(cs, Vm_mV); break;
        case ChannelType::Kv1p1: init_Kv1p1(cs, Vm_mV); break;
        case ChannelType::Kv1p2: init_Kv1p2(cs, Vm_mV); break;
        case ChannelType::Kv1p3: init_Kv1p3(cs, Vm_mV); break;
        case ChannelType::Kv1p4: init_Kv1p4(cs, Vm_mV); break;
        case ChannelType::Kv1p5: init_Kv1p5(cs, Vm_mV); break;
        case ChannelType::Kv1p6: init_Kv1p6(cs, Vm_mV); break;
        case ChannelType::Kv2p1: init_Kv2p1(cs, Vm_mV); break;
        case ChannelType::Kv2p2: init_Kv2p2(cs, Vm_mV); break;
        case ChannelType::Kv3p1: init_Kv3p1(cs, Vm_mV); break;
        case ChannelType::Kv3p2: init_Kv3p2(cs, Vm_mV); break;
        case ChannelType::Kv3p3: init_Kv3p3(cs, Vm_mV); break;
        case ChannelType::Kv3p4: init_Kv3p4(cs, Vm_mV); break;
        case ChannelType::K_Fast: init_K_Fast(cs, Vm_mV); break;
        case ChannelType::KLeak: init_KLeak(cs, Vm_mV); break;
        case ChannelType::Kir2p1: init_Kir2p1(cs, Vm_mV); break;
        case ChannelType::Cav1p2: init_Cav1p2(cs, Vm_mV); break;
        case ChannelType::Cav1p3: init_Cav1p3(cs, Vm_mV); break;
        case ChannelType::Cav2p1: init_Cav2p1(cs, Vm_mV); break;
        case ChannelType::Cav2p2: init_Cav2p2(cs, Vm_mV); break;
        case ChannelType::Cav2p3: init_Cav2p3(cs, Vm_mV); break;
        case ChannelType::Cav3p1: init_Cav3p1(cs, Vm_mV); break;
        case ChannelType::Cav3p3: init_Cav3p3(cs, Vm_mV); break;
        case ChannelType::Ca_L2: init_Ca_L2(cs, Vm_mV); break;
        case ChannelType::Ca_L3: init_Ca_L3(cs, Vm_mV); break;
        case ChannelType::Cav_G: init_Cav_G(cs, Vm_mV); break;
        case ChannelType::CaLeak: init_CaLeak(cs, Vm_mV); break;
        case ChannelType::HCN1: init_HCN1(cs, Vm_mV); break;
        case ChannelType::HCN2: init_HCN2(cs, Vm_mV); break;
        case ChannelType::HCN4: init_HCN4(cs, Vm_mV); break;
        case ChannelType::HCNLeak: init_HCNLeak(cs, Vm_mV); break;
        case ChannelType::HCN2_cAMP: init_HCN2_cAMP(cs, Vm_mV); break;
        case ChannelType::HCN4_cAMP: init_HCN4_cAMP(cs, Vm_mV); break;
        case ChannelType::ClLeak: init_ClLeak(cs, Vm_mV); break;
        case ChannelType::CatLeak: init_CatLeak(cs, Vm_mV); break;
        case ChannelType::CatLeak2: init_CatLeak2(cs, Vm_mV); break;
        case ChannelType::Kv_ML1: init_Kv_ML1(cs, Vm_mV); break;
        case ChannelType::Kv2p1_ML: init_Kv2p1_ML(cs, Vm_mV); break;
        case ChannelType::Kv1p3_ML: init_Kv1p3_ML(cs, Vm_mV); break;
        case ChannelType::Kv1p5_ML: init_Kv1p5_ML(cs, Vm_mV); break;
        case ChannelType::Kv1p5S_ML: init_Kv1p5S_ML(cs, Vm_mV); break;
        case ChannelType::Nav_ML: init_Nav_ML(cs, Vm_mV); break;
        case ChannelType::Cav_L_ML: init_Cav_L_ML(cs, Vm_mV); break;
        case ChannelType::Cav_L_ML2: init_Cav_L_ML2(cs, Vm_mV); break;
        case ChannelType::Cav_N_ML: init_Cav_N_ML(cs, Vm_mV); break;
        case ChannelType::Cav_T_ML: init_Cav_T_ML(cs, Vm_mV); break;
        case ChannelType::Kir_ML: init_Kir_ML(cs, Vm_mV); break;
        case ChannelType::HCN2_ML: init_HCN2_ML(cs, Vm_mV); break;
        case ChannelType::HCN4_ML: init_HCN4_ML(cs, Vm_mV); break;
        case ChannelType::TRP: init_TRP(cs, Vm_mV); break;
    }
}

inline void channel_calc(ChannelState& cs, ChannelType type, const std::vector<double>& Vm_mV) {
    switch (type) {
        case ChannelType::Nav1p2: calc_Nav1p2(cs, Vm_mV); break;
        case ChannelType::Nav1p3: calc_Nav1p3(cs, Vm_mV); break;
        case ChannelType::Nav1p6: calc_Nav1p6(cs, Vm_mV); break;
        case ChannelType::NavRat1: calc_NavRat1(cs, Vm_mV); break;
        case ChannelType::NavRat2: calc_NavRat2(cs, Vm_mV); break;
        case ChannelType::NavRat3: calc_NavRat3(cs, Vm_mV); break;
        case ChannelType::NaLeak: calc_NaLeak(cs, Vm_mV); break;
        case ChannelType::Kv1p1: calc_Kv1p1(cs, Vm_mV); break;
        case ChannelType::Kv1p2: calc_Kv1p2(cs, Vm_mV); break;
        case ChannelType::Kv1p3: calc_Kv1p3(cs, Vm_mV); break;
        case ChannelType::Kv1p4: calc_Kv1p4(cs, Vm_mV); break;
        case ChannelType::Kv1p5: calc_Kv1p5(cs, Vm_mV); break;
        case ChannelType::Kv1p6: calc_Kv1p6(cs, Vm_mV); break;
        case ChannelType::Kv2p1: calc_Kv2p1(cs, Vm_mV); break;
        case ChannelType::Kv2p2: calc_Kv2p2(cs, Vm_mV); break;
        case ChannelType::Kv3p1: calc_Kv3p1(cs, Vm_mV); break;
        case ChannelType::Kv3p2: calc_Kv3p2(cs, Vm_mV); break;
        case ChannelType::Kv3p3: calc_Kv3p3(cs, Vm_mV); break;
        case ChannelType::Kv3p4: calc_Kv3p4(cs, Vm_mV); break;
        case ChannelType::K_Fast: calc_K_Fast(cs, Vm_mV); break;
        case ChannelType::KLeak: calc_KLeak(cs, Vm_mV); break;
        case ChannelType::Kir2p1: calc_Kir2p1(cs, Vm_mV); break;
        case ChannelType::Cav1p2: calc_Cav1p2(cs, Vm_mV); break;
        case ChannelType::Cav1p3: calc_Cav1p3(cs, Vm_mV); break;
        case ChannelType::Cav2p1: calc_Cav2p1(cs, Vm_mV); break;
        case ChannelType::Cav2p2: calc_Cav2p2(cs, Vm_mV); break;
        case ChannelType::Cav2p3: calc_Cav2p3(cs, Vm_mV); break;
        case ChannelType::Cav3p1: calc_Cav3p1(cs, Vm_mV); break;
        case ChannelType::Cav3p3: calc_Cav3p3(cs, Vm_mV); break;
        case ChannelType::Ca_L2: calc_Ca_L2(cs, Vm_mV); break;
        case ChannelType::Ca_L3: calc_Ca_L3(cs, Vm_mV); break;
        case ChannelType::Cav_G: calc_Cav_G(cs, Vm_mV); break;
        case ChannelType::CaLeak: calc_CaLeak(cs, Vm_mV); break;
        case ChannelType::HCN1: calc_HCN1(cs, Vm_mV); break;
        case ChannelType::HCN2: calc_HCN2(cs, Vm_mV); break;
        case ChannelType::HCN4: calc_HCN4(cs, Vm_mV); break;
        case ChannelType::HCNLeak: calc_HCNLeak(cs, Vm_mV); break;
        case ChannelType::HCN2_cAMP: calc_HCN2_cAMP(cs, Vm_mV); break;
        case ChannelType::HCN4_cAMP: calc_HCN4_cAMP(cs, Vm_mV); break;
        case ChannelType::ClLeak: calc_ClLeak(cs, Vm_mV); break;
        case ChannelType::CatLeak: calc_CatLeak(cs, Vm_mV); break;
        case ChannelType::CatLeak2: calc_CatLeak2(cs, Vm_mV); break;
        case ChannelType::Kv_ML1: calc_Kv_ML1(cs, Vm_mV); break;
        case ChannelType::Kv2p1_ML: calc_Kv2p1_ML(cs, Vm_mV); break;
        case ChannelType::Kv1p3_ML: calc_Kv1p3_ML(cs, Vm_mV); break;
        case ChannelType::Kv1p5_ML: calc_Kv1p5_ML(cs, Vm_mV); break;
        case ChannelType::Kv1p5S_ML: calc_Kv1p5S_ML(cs, Vm_mV); break;
        case ChannelType::Nav_ML: calc_Nav_ML(cs, Vm_mV); break;
        case ChannelType::Cav_L_ML: calc_Cav_L_ML(cs, Vm_mV); break;
        case ChannelType::Cav_L_ML2: calc_Cav_L_ML2(cs, Vm_mV); break;
        case ChannelType::Cav_N_ML: calc_Cav_N_ML(cs, Vm_mV); break;
        case ChannelType::Cav_T_ML: calc_Cav_T_ML(cs, Vm_mV); break;
        case ChannelType::Kir_ML: calc_Kir_ML(cs, Vm_mV); break;
        case ChannelType::HCN2_ML: calc_HCN2_ML(cs, Vm_mV); break;
        case ChannelType::HCN4_ML: calc_HCN4_ML(cs, Vm_mV); break;
        case ChannelType::TRP: calc_TRP(cs, Vm_mV); break;
    }
}

// Full channel step: calc state -> update gates -> compute P
inline void channel_step(ChannelState& cs, ChannelType type,
                          const std::vector<double>& Vm_mV, double dt, int total_mems) {
    channel_calc(cs, type, Vm_mV);

    // Determine update method based on channel type
    bool is_ML = (type >= ChannelType::Kv_ML1 && type <= ChannelType::HCN4_ML);
    if (is_ML) {
        if (cs.kinetic_gate) {
            update_ml(cs, dt);
        } else {
            // For non-kinetic ML channels, m = mInf directly
            for (size_t i = 0; i < cs.m.size(); i++) {
                cs.m[i] = cs.mInf_v[i];
            }
        }
    } else {
        update_mh(cs, dt);
    }

    compute_open_prob(cs, total_mems);
}

// ============================================================================
// Active Channel Instance (placed in a tissue region)
// ============================================================================
struct ActiveChannel {
    ChannelType type;
    ChannelState state;
    double max_Dm = 1.0e-15;  // maximum membrane diffusion constant [m/s]
    std::vector<int> target_mems;  // membrane indices this channel applies to
};

} // namespace betse
