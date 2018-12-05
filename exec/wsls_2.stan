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
  vector[2] mu_p;
  vector<lower=0>[2] sig;

  // Subject-level raw parameters (for Matt trick)
  vector[N] ws_r;
  vector[N] ls_r;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N] ws;   // P(stay|win)
  vector<lower=0, upper=1>[N] ls;   // P(shift|loss)

  for (i in 1:N) {
    ws[i] = Phi_approx(mu_p[1] + sig[1] * ws_r[i]);
    ls[i] = Phi_approx(mu_p[2] + sig[2] * ls_r[i]);
  }
}

model {
  // Hyperparameters
  mu_p ~ normal(0, 1.0);
  sig ~ cauchy(0, 5);

  // Individual parameters
  ws_r ~ normal(0, 1.0);
  ls_r ~ normal(0, 1.0);

  for (i in 1:N) {
    vector[4] p;    // probability of choosing each deck
    real prev_outcome;

    p = init;
    prev_outcome = 0;

    for (t in 1:T_subj[i]) {
        choice[i, t] ~ categorical(p);

        if (outcome[i, t] >= prev_outcome) { // win trial
            p = rep_vector((1 - ws[i]) / 3, 4);
            p[choice[i, t]] = ws[i];
        } else {                  // loss trial
            p = rep_vector(ls[i] / 3, 4);
            p[choice[i, t]] = 1 - ls[i];
        }

        prev_outcome = outcome[i, t];
    }
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1> mu_ws;
  real<lower=0, upper=1> mu_ls;

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

  mu_ws = Phi_approx(mu_p[1]);
  mu_ls = Phi_approx(mu_p[2]);
  
  { // local section, this saves time and space
    for (i in 1:N) {
      // Define values
      vector[4] p;
      real prev_outcome;
      
      // Initialize values
      log_lik[i] = 0.0;
      p = init;
      prev_outcome = 0;

      for (t in 1:T_subj[i]) {
        // softmax choice
        log_lik[i] = log_lik[i] + categorical_lpmf(choice[i, t] | p);

        // generate posterior prediction for current trial
        y_pred[i, t] = categorical_rng(p);

        if (outcome[i, t] >= prev_outcome) { // win trial
            p = rep_vector((1 - ws[i]) / 3, 4);
            p[choice[i, t]] = ws[i];
        } else {                  // loss trial
            p = rep_vector(ls[i] / 3, 4);
            p[choice[i, t]] = 1 - ls[i];
        }

        prev_outcome = outcome[i, t];
        
      }
    }
  }
}