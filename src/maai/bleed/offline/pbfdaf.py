"""Linear AEC: dual-filter Partitioned-Block Frequency-Domain Adaptive Filter (offline).

- background filter: always adapts (per-bin normalized NLMS, gradient-constrained overlap-save)
- foreground filter: stable filter used for the output. Copied from the background only when the
  background is judged to have a smaller error (Speex MDF / Beryl stable + adaptive structure);
  restored from the foreground when the background diverges.
- Step size: Valin (2007) optimal step approximation mu_k = leak * |Y_k|^2 / |E_k|^2, which shrinks
  automatically when the near-end talker increases the error. Before convergence a fixed step is
  used, only in blocks where the reference is much louder than the mic (far-end only).
"""
import numpy as np

rfft, irfft = np.fft.rfft, np.fft.irfft


def pbfdaf(mic, ref, block=256, n_part=8, mu_max=0.5, mu_init=0.3,
           dtd_ratio_db=6.0, adapt_sec_needed=3.0, adapt_mode="far_single", near_hold_ms=300.0, sr=16000):
    """Linearly remove the bleed of ``ref`` from ``mic``.

    adapt_mode:
      "far_single": adapt only in blocks where the reference is ``dtd_ratio_db`` louder than the mic
                    (far-end only). The reference (partner's lapel mic) also contains bleed of the
                    near-end voice, so adapting during double talk learns to cancel the near-end voice.
                    Adaptation is also frozen for ``near_hold_ms`` after near-end speech (its reverb
                    tail remains only in the reference and looks like far-end-only speech).
      "valin":      adapt with Valin's variable step except in near-end dominant blocks (closer to
                    a generic AEC).

    Returns:
        dict: e (error = linear AEC output), y (estimated bleed), leak (per block), stats.
    """
    N = block
    K = N + 1
    L = len(mic)
    nblk = int(np.ceil(L / N))
    pad = nblk * N - L
    d_all = np.concatenate([mic, np.zeros(pad)]).astype(np.float64)
    x_all = np.concatenate([ref[:L], np.zeros(pad + max(0, L - len(ref)))])[: nblk * N].astype(np.float64)

    X = np.zeros((n_part, K), np.complex128)  # reference spectrum history (circular buffer)
    Wf = np.zeros((n_part, K), np.complex128)
    Wb = np.zeros((n_part, K), np.complex128)
    x_prev = np.zeros(N)
    zN = np.zeros(N)

    e_out = np.zeros(nblk * N, np.float32)
    y_out = np.zeros(nblk * N, np.float32)
    leak_out = np.zeros(nblk, np.float32)

    # power for normalization
    Psum = np.zeros(K)
    noise_floor = 1e-10 * N
    # leak estimate (Speex MDF)
    Eh = np.zeros(K)
    Yh = np.zeros(K)
    Pey, Pyy = 1.0, 1.0
    leak = 0.25
    # foreground / background switching statistics
    Davg1 = Davg2 = Dvar1 = Dvar2 = 0.0
    adapted_blocks = 0
    adapted = False
    need = int(adapt_sec_needed * sr / N)
    dtd_ratio = 10 ** (dtd_ratio_db / 10)
    n_fg_update = n_bg_reset = 0
    head = 0  # X[head] is the newest
    hold_blocks = int(near_hold_ms * 1e-3 * sr / N)
    hold = 0

    for i in range(nblk):
        sl = slice(i * N, (i + 1) * N)
        d = d_all[sl]
        x = x_all[sl]
        head = (head - 1) % n_part
        X[head] = rfft(np.concatenate([x_prev, x]))
        x_prev = x
        order = (np.arange(n_part) + head) % n_part  # newest first
        Xo = X[order]

        yf = irfft((Wf * Xo).sum(0), 2 * N)[N:]
        yb = irfft((Wb * Xo).sum(0), 2 * N)[N:]
        ef = d - yf
        eb = d - yb

        # --- foreground / background decision (Speex MDF) ---
        Sff = ef @ ef
        See = eb @ eb
        Dbf = (ef - eb) @ (ef - eb) + 1e-12
        diff = Sff - See
        Davg1 = 0.6 * Davg1 + 0.4 * diff
        Dvar1 = 0.36 * Dvar1 + 0.16 * Sff * Dbf
        Davg2 = 0.85 * Davg2 + 0.15 * diff
        Dvar2 = 0.7225 * Dvar2 + 0.0225 * Sff * Dbf
        if (diff * abs(diff) > Sff * Dbf) or (Davg1 * abs(Davg1) > 0.5 * Dvar1) \
                or (Davg2 * abs(Davg2) > 0.25 * Dvar2):
            Wf[:] = Wb
            ef, yf = eb, yb
            Davg1 = Davg2 = Dvar1 = Dvar2 = 0.0
            n_fg_update += 1
        elif -diff * abs(diff) > 4.0 * Sff * Dbf:
            Wb[:] = Wf
            eb, yb = ef, yf
            Davg1 = Davg2 = Dvar1 = Dvar2 = 0.0
            n_bg_reset += 1

        e_out[sl] = ef
        y_out[sl] = yf

        # --- spectra ---
        Ef = np.abs(rfft(np.concatenate([zN, ef]))) ** 2
        Yf = np.abs(rfft(np.concatenate([zN, yf]))) ** 2
        Eb = rfft(np.concatenate([zN, eb]))
        Ebp = np.abs(Eb) ** 2
        Ybp = np.abs(rfft(np.concatenate([zN, yb]))) ** 2

        # --- step size ---
        Xp = (np.abs(Xo) ** 2).sum(0)
        Psum = 0.7 * Psum + 0.3 * Xp
        ed = d @ d
        ex = x @ x
        if ed > dtd_ratio * ex:
            hold = hold_blocks
        elif hold > 0:
            hold -= 1
        far_single = ex > dtd_ratio * ed and ex > noise_floor and hold == 0
        # leak (residual / estimated bleed) from correlation
        Eh_cur = Ef - Eh
        Yh_cur = Yf - Yh
        Eh = 0.95 * Eh + 0.05 * Ef
        Yh = 0.95 * Yh + 0.05 * Yf
        pey = float(Eh_cur @ Yh_cur)
        pyy = float(Yh_cur @ Yh_cur)
        ss = 0.35 / n_part
        if pyy > 0 and far_single:  # during near-end speech E and Y correlate through the near-end bleed
            Pey = (1 - ss) * Pey + ss * pey
            Pyy = (1 - ss) * Pyy + ss * pyy
        if Pyy < 1e-20:
            Pyy = 1e-20
        leak = float(np.clip(Pey / Pyy, 0.005, 1.0)) if adapted else leak
        leak_out[i] = leak

        if not adapted:
            if far_single:
                mu = np.full(K, mu_init)
                adapted_blocks += 1
                if adapted_blocks >= need:
                    adapted = True
            else:
                mu = None
        else:
            mu = np.minimum(mu_max, leak * Ybp / (Ebp + 1e-12))
            # Do not adapt in clearly near-end dominant blocks
            if (adapt_mode == "far_single" and not far_single) or ed > dtd_ratio * ex:
                mu = None
        if mu is not None:
            delta = 1e-6 * (Psum.mean() + 1e-10) + noise_floor
            G = (mu / (Psum + delta)) * Eb
            G = np.conj(Xo) * G[None, :]
            g = irfft(G, 2 * N, axis=1)
            g[:, N:] = 0.0  # gradient constraint (linear convolution)
            Wb += rfft(g, axis=1)  # Wb[p] multiplies the reference p blocks ago

    stats = dict(adapted=adapted, fg_updates=n_fg_update, bg_resets=n_bg_reset,
                 final_leak=leak, median_leak=float(np.median(leak_out[leak_out > 0])) if np.any(leak_out > 0) else None)
    return dict(e=e_out[:L], y=y_out[:L], leak=leak_out, stats=stats)
