data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> T_subj[N];
  real outcome[N, T];
  int<lower=1, upper=4> choice[N, T];
}

transformed data {
  vector[4] init;
  init = rep_vector(0.25, 4);
}

parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[6] mu_p;
  vector<lower=0>[6] sig;

  // Subject-level raw parameters (for Matt trick)
  vector[N] ws_init_r;
  vector[N] ls_init_r;
  vector[N] theta_ws_r;
  vector[N] theta_ls_r;
  vector[N] ws_fin_r;
  vector[N] ls_fin_r;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N] ws_init;  // initial P(stay|win)
  vector<lower=0, upper=1>[N] ls_init;  // initial P(shift|loss)
  vector<lower=0, upper=1>[N] theta_ws; // change rate of P(stay|win) on each trial
  vector<lower=0, upper=1>[N] theta_ls; // change rate of P(shift|loss) on each trial
  vector<lower=0, upper=1>[N] ws_fin;   // asymptotic ending value of P(stay|win)
  vector<lower=0, upper=1>[N] ls_fin;   // asymptotic ending value of P(shift|loss)

  for (i in 1:N) {
    ws_init[i] = Phi_approx(mu_p[1] + sig[1] * ws_init_r[i]);
    ls_init[i] = Phi_approx(mu_p[2] + sig[2] * ls_init_r[i]);
    theta_ws[i] = Phi_approx(mu_p[3] + sig[3] * theta_ws_r[i]);
    theta_ls[i] = Phi_approx(mu_p[4] + sig[4] * theta_ls_r[i]);
    ws_fin[i] = Phi_approx(mu_p[5] + sig[5] * ws_fin_r[i]);
    ls_fin[i] = Phi_approx(mu_p[6] + sig[6] * ws_fin_r[i]);
  }
}

model {
  // Hyperparameters
  mu_p ~ normal(0, 1.0);
  sig ~ cauchy(0, 5);

  // Individual parameters
  ws_init_r ~ normal(0, 1.0);
  ls_init_r ~ normal(0, 1.0);
  theta_ws_r ~ normal(0, 1.0);
  theta_ls_r ~ normal(0, 1.0);
  ws_fin_r ~ normal(0, 1.0);
  ls_fin_r ~ normal(0, 1.0);

  for (i in 1:N) {
    real ws;       // P(stay|win)
    real ls;       // P(shift|loss)
    vector[4] p;   // probability of choosing each option
    real prev_outcome;

    ws = ws_init[i];
    ls = ls_init[i];
    p  = init;
    prev_outcome = 0;

    for (t in 1:T_subj[i]) {
        choice[i, t] ~ categorical(p);

        if (outcome[i, t] >= prev_outcome) {  // win trial
            ws = ws + theta_ws[i] * (ws_fin[i] - ws);
            p = rep_vector((1 - ws) / 3, 4);
            p[choice[i, t]] = ws;
        } else {                   // loss trial
            ls = ls + theta_ls[i] * (ls_fin[i] - ls);
            p = rep_vector(ls / 3, 4);
            p[choice[i, t]] = 1 - ls;
        } 

        prev_outcome = choice[i, t];
    }
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1> mu_ws_init;
  real<lower=0, upper=1> mu_ls_init;
  real<lower=0, upper=1> mu_theta_ws;
  real<lower=0, upper=1> mu_theta_ls; 
  real<lower=0, upper=1> mu_ws_fin;
  real<lower=0, upper=1> mu_ls_fin;

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

  mu_ws_init = Phi_approx(mu_p[1]);
  mu_ls_init = Phi_approx(mu_p[2]);
  mu_theta_ws = Phi_approx(mu_p[3]);
  mu_theta_ls = Phi_approx(mu_p[4]);
  mu_ws_fin = Phi_approx(mu_p[5]);
  mu_ls_fin = Phi_approx(mu_p[6]);
  
  { // local section, this saves time and space
    for (i in 1:N) {
      // Define values
      real ws;      // P(stay|win)
      real ls;      // P(shift|loss)
      vector[4] p;  // probability of choosing each option
      real prev_outcome;
      
      // Initialize values
      log_lik[i] = 0;
      ws = ws_init[i];
      ls = ls_init[i];
      p = init;
      prev_outcome = 0;

      for (t in 1:T_subj[i]) {
        
        log_lik[i] = log_lik[i] + categorical_lpmf(choice[i, t] | p);

        // generate posterior prediction for current trial
        y_pred[i, t] = categorical_rng(p);

        if (outcome[i, t] >= prev_outcome) {  // win trial
            ws = ws + theta_ws[i] * (ws_fin[i] - ws);
            p = rep_vector((1 - ws) / 3, 4);
            p[choice[i, t]] = ws;
        } else {                   // loss trial
            ls = ls + theta_ls[i] * (ls_fin[i] - ls);
            p = rep_vector(ls / 3, 4);
            p[choice[i, t]] = 1 - ls;
        } 

        prev_outcome = outcome[i, t];
        
      }
    }
  }
}