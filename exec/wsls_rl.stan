data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> T_subj[N];
  real outcome[N, T];
  int<lower=1, upper=4> choice[N, T];
}

transformed data {
  vector[4] init_0;
  vector[4] init_p;
  init_0 = rep_vector(0.0, 4);
  init_p = rep_vector(0.25, 4);
}

parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[13] mu_p;
  vector<lower=0>[13] sig;

  // Subject-level raw parameters (for Matt trick)
  vector[N] alpha_r;
  vector[N] gamma_r;
  vector[N] ev_init_r;
  vector[N] zeta_r;
  vector[N] lambda_r;
  vector[N] ep_r;
  vector[N] ws_init_r;
  vector[N] ls_init_r;
  vector[N] theta_ws_r;
  vector[N] theta_ls_r;
  vector[N] ws_fin_r;
  vector[N] ls_fin_r;
  vector[N] k_r;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N] alpha;    // learning rate (i.e. recency)
  vector<lower=0, upper=15>[N] gamma;   // exploitation
  vector[N] ev_init;                    // initial EVs for all options
  vector<lower=0, upper=1>[N] zeta;     // eligibility trace decay
  vector<lower=0, upper=1>[N] lambda;   // EV decay
  vector[N] ep;                         // EV baseline
  vector<lower=0, upper=1>[N] ws_init;  // initial P(stay|win)
  vector<lower=0, upper=1>[N] ls_init;  // initial P(shift|loss)
  vector<lower=0, upper=1>[N] theta_ws; // change rate of P(stay|win) on each trial
  vector<lower=0, upper=1>[N] theta_ls; // change rate of P(shift|loss) on each trial
  vector<lower=0, upper=1>[N] ws_fin;   // asymptotic ending value of P(stay|win)
  vector<lower=0, upper=1>[N] ls_fin;   // asymptotic ending value of P(shift|loss)
  vector<lower=0, upper=1>[N] k;        // interpolation weight


  for (i in 1:N) {
    alpha[i]  = Phi_approx(mu_p[1] + sig[1] * alpha_r[i]);
    gamma[i] = Phi_approx(mu_p[2] + sig[2] * gamma_r[i]) * 15;
    zeta[i]   = Phi_approx(mu_p[4] + sig[4] * zeta_r[i]);
    lambda[i] = Phi_approx(mu_p[5] + sig[5] * k_r[i]);
    ws_init[i] = Phi_approx(mu_p[7] + sig[7] * ws_init_r[i]);
    ls_init[i] = Phi_approx(mu_p[8] + sig[8] * ls_init_r[i]);
    theta_ws[i] = Phi_approx(mu_p[9] + sig[9] * theta_ws_r[i]);
    theta_ls[i] = Phi_approx(mu_p[10] + sig[10] * theta_ls_r[i]);
    ws_fin[i] = Phi_approx(mu_p[11] + sig[11] * ws_fin_r[i]);
    ls_fin[i] = Phi_approx(mu_p[12] + sig[12] * ws_fin_r[i]);
    k[i]      = Phi_approx(mu_p[13] + sig[13] * k_r[i]);
  }
  ev_init = mu_p[3] + sig[3] * ev_init_r;
  ep = mu_p[6] + sig[6] * ep_r;
}

model {
  // Hyperparameters
  mu_p[1]  ~ normal(0, 1.0);
  mu_p[2]  ~ normal(0, 1.0);
  mu_p[3]  ~ normal(0, 10.0);
  mu_p[4]  ~ normal(0, 1.0);
  mu_p[5]  ~ normal(0, 1.0);
  mu_p[6]  ~ normal(0, 10.0);
  mu_p[7]  ~ normal(0, 1.0);
  mu_p[8]  ~ normal(0, 1.0);
  mu_p[9]  ~ normal(0, 1.0);
  mu_p[10] ~ normal(0, 1.0);
  mu_p[11] ~ normal(0, 1.0);
  mu_p[12] ~ normal(0, 1.0);
  mu_p[13] ~ normal(0, 1.0);
  sig ~ cauchy(0, 5.0);

  // Individual parameters
  alpha_r ~ normal(0, 1.0);
  gamma_r ~ normal(0, 1.0);
  ev_init_r ~ normal(0, 1.0);
  zeta_r ~ normal(0, 1.0);
  lambda_r ~ normal(0, 1.0);
  ep_r ~ normal(0, 1.0);
  ws_init_r ~ normal(0, 1.0);
  ls_init_r ~ normal(0, 1.0);
  theta_ws_r ~ normal(0, 1.0);
  theta_ls_r ~ normal(0, 1.0);
  ws_fin_r ~ normal(0, 1.0);
  ls_fin_r ~ normal(0, 1.0);
  k_r ~ normal(0, 1.0);

  for (i in 1:N) {
    // Define values
    vector[4] ev;
    vector[4] elig;   // eligibility trace
    vector[4] p_rl;   // RL element of determining p
    vector[4] p_wsls; // WSLS element of determining p
    vector[4] p;      // probability of choosing each option with dual strategy
    real ws;          // P(stay|win)
    real ls;          // P(shift|loss)
    real prev_outcome;

    // Initialize values
    ev    = init_0;
    elig  = init_0;
    p_rl  = init_p;
    p_wsls = init_p;
    p     = init_p;
    ws    = ws_init[i];
    ls    = ls_init[i];
    prev_outcome = 0;

    for (t in 1:T_subj[i]) {
      // choice with dual strategy
      choice[i, t] ~ categorical(p);

      // RL part

      // eligibility trace increment
      elig[choice[i, t]] = elig[choice[i, t]] + 1;
      // EV update
      for(j in 1:4) {
        ev[j] = ev[j] + alpha[i] * (outcome[i, t] - ev[j]) * elig[j];
      }
      // eligibility trace decay
      elig = elig * zeta[i];
      // EV decay
      ev = lambda[i] * ev + (1 - lambda[i]) * ep[i];
      // RL probability
      p_rl = softmax(gamma[i] * ev);

      // WSLS part
      if (outcome[i, t] >= prev_outcome) {  // win trial
        ws = ws + theta_ws[i] * (ws_fin[i] - ws);
        p_wsls = rep_vector((1 - ws) / 3, 4);
        p_wsls[choice[i, t]] = ws;
      } else {                   // loss trial
        ls = ls + theta_ls[i] * (ls_fin[i] - ls);
        p_wsls = rep_vector(ls / 3, 4);
        p_wsls[choice[i, t]] = 1 - ls;
      } 

      // Interpolation of WSLS and RL
      p = k[i] * p_wsls + (1 - k[i]) * p_rl;

      // Update previous outcome
      prev_outcome = outcome[i, t];
    }
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1> mu_alpha;
  real<lower=0> mu_gamma;
  real mu_ev_init;
  real<lower=0, upper=1> mu_zeta;
  real<lower=0, upper=1> mu_lambda;
  real mu_ep; 
  real<lower=0, upper=1> mu_ws_init;
  real<lower=0, upper=1> mu_ls_init;
  real<lower=0, upper=1> mu_theta_ws;
  real<lower=0, upper=1> mu_theta_ls; 
  real<lower=0, upper=1> mu_ws_fin;
  real<lower=0, upper=1> mu_ls_fin;
  real<lower=0, upper=1> mu_k;

  // For log likelihood calculation
  real log_lik[N];

  // For posterior predictive check
  real y_pred[N, T];

  // Set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_alpha  = Phi_approx(mu_p[1]);
  mu_gamma = Phi_approx(mu_p[2]) * 15;
  mu_ev_init = mu_p[3];
  mu_zeta = Phi_approx(mu_p[4]);
  mu_lambda = Phi_approx(mu_p[5]);
  mu_ep = mu_p[6];
  mu_ws_init = Phi_approx(mu_p[7]);
  mu_ls_init = Phi_approx(mu_p[8]);
  mu_theta_ws = Phi_approx(mu_p[9]);
  mu_theta_ls = Phi_approx(mu_p[10]);
  mu_ws_fin = Phi_approx(mu_p[11]);
  mu_ls_fin = Phi_approx(mu_p[12]);
  mu_k = Phi_approx(mu_p[13]);


  { // local section, this saves time and space
    for (i in 1:N) {
      // Define values
      vector[4] ev;
      vector[4] elig;   // eligibility trace
      vector[4] p_rl;   // RL element of determining p
      vector[4] p_wsls; // WSLS element of determining p
      vector[4] p;      // probability of choosing each option with dual strategy
      real ws;          // P(stay|win)
      real ls;          // P(shift|loss)
      real prev_outcome;

      // Initialize values
      log_lik[i] = 0;
      ev = init_0;
      elig = init_0;
      p_rl = init_p;
      p_wsls = init_p;
      p = init_p;
      ws = ws_init[i];
      ls = ls_init[i];
      prev_outcome = 0;

      for (t in 1:T_subj[i]) {

        // choice with dual strategy
        log_lik[i] = log_lik[i] + categorical_lpmf(choice[i, t] | p);

        // generate posterior prediction for current trial
        y_pred[i, t] = categorical_rng(p);

        // RL part

        // eligibility trace increment
        elig[choice[i, t]] = elig[choice[i, t]] + 1;
        // EV update
        for(j in 1:4) {
          ev[j] = ev[j] + alpha[i] * (outcome[i, t] - ev[j]) * elig[j];
        }
        // eligibility trace decay
        elig = elig * zeta[i];
        // EV decay
        ev = lambda[i] * ev + (1 - lambda[i]) * ep[i];
        // RL probability
        p_rl = softmax(gamma[i] * ev);

        // WSLS part
        if (outcome[i, t] >= prev_outcome) {  // win trial
          ws = ws + theta_ws[i] * (ws_fin[i] - ws);
          p_wsls = rep_vector((1 - ws) / 3, 4);
          p_wsls[choice[i, t]] = ws;
        } else {                   // loss trial
          ls = ls + theta_ls[i] * (ls_fin[i] - ls);
          p_wsls = rep_vector(ls / 3, 4);
          p_wsls[choice[i, t]] = 1 - ls;
        } 

        // Interpolation of WSLS and RL
        p = k[i] * p_wsls + (1 - k[i]) * p_rl;

        // Update previous outcome
        prev_outcome = outcome[i, t];

      }
    }
  }
}