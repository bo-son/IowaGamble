data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> T_subj[N];
  real outcome[N, T];
  int<lower=1, upper=4> choice[N, T];
}

transformed data {
  vector[4] init;
  init = rep_vector(0.0, 4);
}

parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[8] mu_p;
  vector<lower=0>[8] sig;

  // Subject-level raw parameters (for Matt trick)
  vector[N] alpha_r;
  vector[N] lambda_r;
  vector[N] phi_r;
  vector[N] k_r;
  vector[N] ep_pos_r;
  vector[N] ep_neg_r;
  vector[N] w_r;
  vector[N] c_r;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N] alpha;    // shape of utility
  vector<lower=0, upper=5>[N] lambda;  // loss aversion
  vector<lower=0, upper=1>[N] phi;      // recency
  vector<lower=0, upper=1>[N] k;        // perservation decay
  vector[N] ep_pos;                     // perseveration bias after pos
  vector[N] ep_neg;                     // perseveration bias after neg
  vector<lower=0, upper=1>[N] w;        // interpolation weight
  vector<lower=0, upper=5>[N] c;        // response consistency (i.e. exploitation)


  for (i in 1:N) {
    alpha[i]  = Phi_approx(mu_p[1] + sig[1] * alpha_r[i]);
    lambda[i] = Phi_approx(mu_p[2] + sig[2] * lambda_r[i]) * 5;
    phi[i]    = Phi_approx(mu_p[3] + sig[3] * phi_r[i]);
    k[i]      = Phi_approx(mu_p[4] + sig[4] * k_r[i]);
    w[i]      = Phi_approx(mu_p[7] + sig[7] * w_r[i]);
    c[i]      = Phi_approx(mu_p[8] + sig[8] * c_r[i]) * 5;
  }
  ep_pos = mu_p[5] + sig[5] * ep_pos_r;
  ep_neg = mu_p[6] + sig[6] * ep_neg_r;
}

model {
  // Hyperparameters
  mu_p[1]  ~ normal(0, 1.0);
  mu_p[2]  ~ normal(0, 1.0);
  mu_p[3]  ~ normal(0, 1.0);
  mu_p[4]  ~ normal(0, 1.0);
  mu_p[5]  ~ normal(0, 10.0);
  mu_p[6]  ~ normal(0, 10.0);
  mu_p[7]  ~ normal(0, 1.0);
  mu_p[8]  ~ normal(0, 1.0);
  sig ~ cauchy(0, 5);

  // Individual parameters
  alpha_r  ~ normal(0, 1.0);
  lambda_r ~ normal(0, 1.0);
  phi_r    ~ normal(0, 1.0);
  ep_pos_r ~ normal(0, 1.0);
  ep_neg_r ~ normal(0, 1.0);
  k_r      ~ normal(0, 1.0);
  w_r      ~ normal(0, 1.0);
  c_r      ~ normal(0, 1.0);

  for (i in 1:N) {
    // Define values
    vector[4] ev;
    vector[4] pers;   // perseverance
    vector[4] val;   // weighted sum of ev and pers

    real util;     // utility of curFb
    real theta;       // theta = 3^c - 1

    // Initialize valuesd
    theta = pow(3, c[i]) - 1;
    ev    = init; // initial ev values
    pers  = init; // initial pers values
    val   = init;

    for (t in 1:T_subj[i]) {
      // softmax choice
      choice[i, t] ~ categorical_logit(theta * val);

      // perseverance decay
      pers = pers * k[i]; // decay

      if (outcome[i, t] >= 0) {  // x(t) >= 0
        util = pow(outcome[i, t], alpha[i]);
        pers[choice[i, t]] = pers[choice[i, t]] + ep_pos[i];  // perseverance term
      } else {                  // x(t) < 0
        util = -1 * lambda[i] * pow(-1 * outcome[i, t], alpha[i]);
        pers[choice[i, t]] = pers[choice[i, t]] + ep_neg[i];  // perseverance term
      }

      ev[choice[i, t]] = ev[choice[i, t]] + phi[i] * (util - ev[choice[i, t]]);
      // calculate val
      val = w[i] * ev + (1-w[i]) * pers;
    }
  }
}
generated quantities {
  // For group level parameters
  /* TODO: WHAT??? */
  real<lower=0, upper=1>  mu_alpha;
  real<lower=0, upper=5> mu_lambda;
  real<lower=0, upper=1>  mu_phi;
  real mu_ep_pos;
  real mu_ep_neg;
  real<lower=0, upper=1> mu_k;
  real<lower=0, upper=1> mu_w;
  real<lower=0, upper=5> mu_c;

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
  mu_lambda = Phi_approx(mu_p[2]) * 5;
  mu_phi    = Phi_approx(mu_p[3]);
  mu_k      = Phi_approx(mu_p[4]);
  mu_ep_pos = mu_p[5];
  mu_ep_neg = mu_p[6];
  mu_w      = Phi_approx(mu_p[7]);
  mu_c      = Phi_approx(mu_p[8]) * 5;

  { // local section, this saves time and space
    for (i in 1:N) {
      // Define values
      vector[4] ev;
      vector[4] pers;   // perseverance
      vector[4] val;   // weighted sum of ev and pers

      real util;     // utility of curFb
      real theta;       // theta = 3^c - 1

      // Initialize values
      log_lik[i] = 0;
      theta      = pow(3, c[i]) -1;
      ev         = init; // initial ev values
      pers       = init; // initial pers values
      val        = init;

      for (t in 1:T_subj[i]) {
        // softmax choice
        log_lik[i] = log_lik[i] + categorical_logit_lpmf(choice[i, t] | theta * val);

        // generate posterior prediction for current trial
        y_pred[i, t] = categorical_logit_rng(theta * val);
        #y_pred[i, t] = categorical_rng(softmax(theta * val));

        // perseverance decay
        pers = pers * k[i]; // decay

        if (outcome[i, t] >= 0) {  // x(t) >= 0
          util = pow(outcome[i, t], alpha[i]);
          pers[choice[i, t]] = pers[choice[i, t]] + ep_pos[i];  // perseverance term
        } else {                  // x(t) < 0
          util = -1 * lambda[i] * pow(-1 * outcome[i, t], alpha[i]);
          pers[choice[i, t]] = pers[choice[i, t]] + ep_neg[i];  // perseverance term
        }

        ev[choice[i, t]] = ev[choice[i, t]] + phi[i] * (util - ev[choice[i, t]]);
        // calculate V
        val = w[i] * ev + (1 - w[i]) * pers;
      }
    }
  }
}

