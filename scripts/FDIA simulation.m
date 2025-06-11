%% Stealthy FDIA Attack Simulation to Manipulate Locational Marginal Prices
% This script simulates Case 1: a transmission line rating attack by reducing
% the thermal limit (RATE_A) of a selected transmission line during specific weeks.
% You can easily modify this script to simulate other stealthy FDIA types.
%
%% NOTE:
% This script relies on the publicly available NPCC 140-bus case from:
% https://github.com/enliten/ENLITEN-Grid-Econ-Data
%
% Please download both `NPCC.m` and `NPCC_load.csv` from the following path:
% https://github.com/enliten/ENLITEN-Grid-Econ-Data
%
% Make sure both files are placed in your MATLAB working directory or added to your path.

clear; clc;

%% --- USER CONFIGURATION ---

% MATPOWER case file (must be in MATLAB path or folder)
mpc_file = 'NPCC.m';

% Load profile (CSV file: buses x time)
load_file = 'NPCC_load.csv';

% Output file (raw LMP results)
output_file = 'case1_raw.csv';

% Attack settings
target_branch     = 109;             % Branch to attack
weeks_to_attack   = [5, 14, 17];     % Week numbers during which the attack is active
attack_magnitude  = 0.14;            % Reduction in RATE_A (e.g., 0.14 = 14%)

% Time range for simulation (in hours)
t_start = 337;
t_end   = 3361;

%% --- INITIALIZATION ---

define_constants;                             % Load MATPOWER constants
mpc = loadcase(mpc_file);                     % Load system model
load_data = readmatrix(load_file, 'Range', 'A2');  % Load demand profile
[n_buses, ~] = size(load_data);               % Get number of buses

% Define attack target
original_rate = mpc.branch(target_branch, RATE_A);  % Save original RATE_A
attackedLMP = [];                                   % Initialize result matrix

%% --- SIMULATION LOOP ---

for t = t_start:t_end
    current_week = ceil(t / 168);
    attack_flag = 0;

    % Apply attack if current week is in the attack window
    if ismember(current_week, weeks_to_attack)
        mpc.branch(target_branch, RATE_A) = original_rate * (1 - attack_magnitude);
        attack_flag = 1;
    else
        mpc.branch(target_branch, RATE_A) = original_rate;
    end

    % Apply load for this hour
    for b = 1:n_buses
        mpc.bus(b, PD) = load_data(b, t);
    end

    % Run OPF and collect LMPs
    try
        result = runopf(mpc);
        LMP_row = [t; attack_flag; result.bus(:, LAM_P)];
    catch
        warning('OPF failed at t = %d. Filling with NaNs.', t);
        LMP_row = [t; attack_flag; NaN(n_buses, 1)];
    end

    attackedLMP = [attackedLMP, LMP_row];

    % Show progress every 24 hours
    if mod(t, 24) == 0
        fprintf(' Processed hour %d of %d\n', t, t_end);
    end
end

%% --- SAVE OUTPUT ---

writematrix(attackedLMP, output_file);
fprintf('LMP results saved to: %s\n', output_file);
