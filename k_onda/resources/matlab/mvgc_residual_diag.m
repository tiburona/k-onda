function out = mvgc_residual_diag(E, nlags)
%MVGC_RESIDUAL_DIAG Simple per-channel residual autocorrelation diagnostics.
%
% E: residuals (nvars x nobs x ntrials) from tsdata_to_var
% Returns:
%   out.pvals: nvars x 1 (approx chi-square p-values)
%   out.Q:     nvars x 1 (Ljung-Box-ish statistic)
%   out.ok:    boolean
%
% Notes:
%   - Flattens trials by concatenation.
%   - Checks autocorr of residuals; large autocorr => p too small.

    if nargin < 2 || isempty(nlags), nlags = 20; end

    out = struct();
    out.ok = true;

    nvars = size(E,1);
    if ndims(E) == 3
        T = size(E,2);
        N = size(E,3);
        R = reshape(permute(E,[1 2 3]), nvars, T*N); % concat trials
    else
        R = E;
    end

    Ttot = size(R,2);
    out.nlags = nlags;
    out.T = Ttot;

    Q = nan(nvars,1);
    p = nan(nvars,1);

    for i = 1:nvars
        r = R(i,:) - mean(R(i,:));
        % autocorrelation up to nlags
        ac = nan(nlags,1);
        denom = sum(r.^2);
        for k = 1:nlags
            ac(k) = (r(1+k:end) * r(1:end-k)') / denom;
        end

        % Ljung-Box-ish statistic
        % Q = T(T+2) sum_{k=1..m} ac(k)^2/(T-k)
        Q(i) = Ttot*(Ttot+2) * sum( (ac.^2) ./ (Ttot - (1:nlags)') );

        % p-value ~ chi2 with nlags dof
        p(i) = 1 - chi2cdf(Q(i), nlags);
    end

    out.Q = Q;
    out.pvals = p;
end
