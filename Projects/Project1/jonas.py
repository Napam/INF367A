class GibbsSamplerARD():
    '''
    TODO: Docs
    '''
    
    def __init__(self, a0: float, b0: float, c0: float, d0: float):
        '''Initializes a Gibbs sampler for the ARD regression model with a given set of hyperparameters.
        TODO: Docs
        '''
        self.a0 = a0
        self.b0 = b0
        self.c0 = c0
        self.d0 = d0
    
    def __sample_weights__(self, X: np.ndarray, y: np.ndarray, beta: float, alphas: np.ndarray):
        '''
        TODO: Docs
        '''
        # Compute mean (m) and covariance (S) parameters
        D = np.diag(alphas)
        S = np.linalg.inv(beta * X.T @ X + D)
        d = X.shape[1]
        m = (beta * S @ X.T @ y).reshape(d,)
        
        # Sample from multivariate gaussian distribution
        return mvn.rvs(mean=m, cov=S).reshape(-1, 1)
    
    def __sample_alphas__(self, weights: np.ndarray):
        '''
        TODO: Docs
        '''
        d = len(weights)
        alphas = np.zeros(d)
        for i in range(d):
            
            # Compute scale (a) and inverse scale (b) parameters
            a = self.c0 + (1/2)
            b = self.d0 + (1/2) * weights[i]**2
            
            # Sample from gamma distribution
            alphas[i] = gamma.rvs(a=a, scale=1/b)
            
        return alphas
    
    def __sample_beta__(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
        '''
        TODO: Docs
        '''
        # Compute scale (a) and inverse scale (b) parameters
        n = X.shape[0]
        a = self.a0 + n/2
        y_res = y - X @ weights # Residuals
        b = (self.b0 + (1/2) * y_res.T @ y_res)[0, 0]
        
        # Sample from gamma distribution
        return gamma.rvs(a=a, scale=1/b)
    
    def __calc_log_posterior__(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, alphas: np.ndarray, beta: float):
        '''
        TODO: Docs
        '''
        n, d = X.shape
        
        # Calculating the log posterior for y, weights, alphas and beta
        log_posterior = 0
        
        # Log posterior for y
        for i in range(n):
            x_i = X[i].reshape(-1, 1)
            log_posterior += norm.logpdf(y[i], loc=weights.T @ x_i, scale=np.sqrt(1/beta))
        
        # Log posterior for weights
        w_prob = 1
        for j in range(d):
            w_prob *= norm.logpdf(weights[j], loc=0, scale=np.sqrt(1/alphas[j]))
        log_posterior += w_prob
        
        # Log posterior for alphas
        a_prob = 1
        for j in range(d):
            a_prob *= gamma.logpdf(alphas[j], a=self.c0, scale=1/self.d0)
        log_posterior += a_prob
        
        # Log posterior for beta
        log_posterior += gamma.logpdf(beta, a=self.a0, scale=1/self.b0)
        
        return log_posterior[0, 0]
    
    def __calc_log_posterior_data__(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, beta: float):
        '''
        TODO: Docs
        '''
        n, d = X.shape

        # Log posterior for y
        log_posterior = 0
        for i in range(n):
            x_i = X[i].reshape(-1, 1)
            log_posterior += norm.logpdf(y[i], loc=weights.T @ x_i, scale=np.sqrt(1/beta))[0]

        return log_posterior
    
    def __num_params__(self, d: int):
        '''
        TODO: Docs
        '''
        # Beta requires one parameter, and alpha/weights both
        # require d parameters (thus 2 * d).
        return 2 * d + 1
    
    def __bic__(self, lp: float, n: int, d: int):
        '''
        TODO: Docs
        '''
        num_params = self.__num_params__(d)
        return lp - (num_params/2) * np.log(n)
    
    def __aic__(self, lp: float, n: int, d: int):
        '''
        TODO: Docs
        '''
        num_params = self.__num_params__(d)
        return (-2/n) * lp + 2 * (num_params/n)
    
    def __get_map_params__(self, samples: np.ndarray):
        '''
        TODO: Docs
        '''
        d = samples.shape[1] if len(samples.shape) > 1 else 1
        map_params = np.zeros(d)
        for i in range(d):
            samples_i = samples[:, i] if len(samples.shape) > 1 else samples[i]
            hist, bin_edges = np.histogram(samples_i, bins='auto')
            samples_i_argmax = bin_edges[hist.argmax() + 1]
            map_params[i] = samples_i_argmax
        
        return map_params
    
    def fit(self, X: np.ndarray, y: np.ndarray, n_iters: int = 1000, burn_in_perc: float = 0.5):
        '''
        TODO: Docs
        '''
        n, d = X.shape
        y_mat = y.reshape(-1, 1)
        
        # Initialize parameters
        beta = 1
        alphas = np.repeat(1, d)
        
        # Initialize result
        self.weight_samples = np.zeros((n_iters, d))
        self.alpha_samples = np.zeros((n_iters, d))
        self.beta_samples = np.zeros(n_iters)
        self.log_posteriors = np.zeros(n_iters)
        
        # Perform Gibbs sampling
        for i in range(n_iters):
            
            # Sample parameters
            weights = self.__sample_weights__(X, y_mat, beta, alphas)
            alphas = self.__sample_alphas__(weights)
            beta = self.__sample_beta__(X, y_mat, weights)
            
            # Calculate unnormalized log-posterior
            lp = self.__calc_log_posterior__(X, y_mat, weights, alphas, beta)
            
            # Report back results
            self.weight_samples[i] = weights.reshape(d,)
            self.alpha_samples[i] = alphas
            self.beta_samples[i] = beta
            self.log_posteriors[i] = lp
        
        self.ws = self.weight_samples

        # Find best parameters
        weight_hat = self.__get_map_params__(self.weight_samples)
        alpha_hat = self.__get_map_params__(self.alpha_samples)
        beta_hat = self.__get_map_params__(self.beta_samples)
        self.best_params = {
            'weights': weight_hat,
            'alphas': alpha_hat,
            'beta': beta_hat
        }
        
        # Assess performance
        lp_hat = self.__calc_log_posterior_data__(X, y_mat, weight_hat, beta_hat)
        bic = self.__bic__(lp_hat, n, d)
        aic = self.__aic__(lp_hat, n, d)
        self.performance = np.array([bic, aic])
        
        # Calculate burn-in
        self.burn_in = int(n_iters * burn_in_perc)
        
    def get_results(self):
        '''
        TODO: Docs
        '''
        
        # Create lists of samples and corresponding names
        param_samples = []
        param_names = []
        d = self.weight_samples.shape[1]
        
        # Add weight samples/names
        for i in range(d):
            param_samples.append(self.weight_samples[:, i])
            
            suffix = ''
            if i == 0:
                suffix = ' (intercept)'
            elif i == 1:
                suffix = ' (slope)'
            param_names.append(f'w{i}{suffix}')
        
        # Add alpha samples/names
        for i in range(d):
            param_samples.append(self.alpha_samples[:, i])
            param_names.append(f'α{i}')
        
        # Add beta samples/names
        param_samples.append(self.beta_samples)
        param_names.append('β')
        
        return param_samples, param_names


class GibbsSampler():
    '''
    TODO: Docs
    '''
    
    def __init__(self, a0: float, b0: float, alpha: float):
        '''Initializes a standard Gibbs sampler with a given set of hyperparameters.
        TODO: Docs
        '''
        self.a0 = a0
        self.b0 = b0
        self.alpha = alpha
    
    def __sample_weights__(self, X: np.ndarray, y: np.ndarray, beta: float):
        '''
        TODO: Docs
        '''
        # Compute mean (m) and covariance (S) parameters
        d = X.shape[1]
        D = self.alpha * np.eye(d)
        S = np.linalg.inv(beta * X.T @ X + D)
        m = (beta * S @ X.T @ y).reshape(d,)
        
        # Sample from multivariate gaussian distribution
        return mvn.rvs(mean=m, cov=S).reshape(-1, 1)
    
    def __sample_beta__(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
        '''
        TODO: Docs
        '''
        # Compute scale (a) and inverse scale (b) parameters
        n = X.shape[0]
        a = self.a0 + n/2
        y_res = y - X @ weights # Residuals
        b = (self.b0 + (1/2) * y_res.T @ y_res)[0, 0]
        
        # Sample from gamma distribution
        return gamma.rvs(a=a, scale=1/b)
    
    def __calc_log_posterior__(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, beta: float):
        '''
        TODO: Docs
        '''
        n, d = X.shape
        
        # Calculating the log posterior for y, weights and beta
        log_posterior = 0
        
        # Log posterior for y
        # TODO: Replace with multivariate gaussian?
        for i in range(n):
            x_i = X[i].reshape(-1, 1)
            log_posterior += norm.logpdf(y[i], loc=weights.T @ x_i, scale=np.sqrt(1/beta))
        
        # Log posterior for weights
        log_posterior += mvn.logpdf(weights.reshape(d,), mean=np.zeros(d), cov=1/alpha*np.eye(d))
        
        # Log posterior for beta
        log_posterior += gamma.logpdf(beta, a=self.a0, scale=1/self.b0)
        
        return log_posterior[0, 0]
    
    def __calc_log_posterior_data__(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, beta: float):
        '''
        TODO: Docs
        '''
        n, d = X.shape

        # Log posterior for y
        log_posterior = 0
        for i in range(n):
            x_i = X[i].reshape(-1, 1)
            log_posterior += norm.logpdf(y[i], loc=weights.T @ x_i, scale=np.sqrt(1/beta))[0]

        return log_posterior
    
    def __num_params__(self, d: int):
        '''
        TODO: Docs
        '''
        # Beta requires one parameter and weights require d parameters
        return d + 1
    
    def __bic__(self, lp: float, n: int, d: int):
        '''
        TODO: Docs
        '''
        num_params = self.__num_params__(d)
        return lp - (num_params/2) * np.log(n)
    
    def __aic__(self, lp: float, n: int, d: int):
        '''
        TODO: Docs
        '''
        num_params = self.__num_params__(d)
        return (-2/n) * lp + 2 * (num_params/n)
    
    def __get_map_params__(self, samples: np.ndarray):
        '''
        TODO: Docs
        '''
        d = samples.shape[1] if len(samples.shape) > 1 else 1
        map_params = np.zeros(d)
        for i in range(d):
            samples_i = samples[:, i] if len(samples.shape) > 1 else samples[i]
            hist, bin_edges = np.histogram(samples_i, bins='auto')
            samples_i_argmax = bin_edges[hist.argmax() + 1]
            map_params[i] = samples_i_argmax
        
        return map_params
    
    def fit(self, X: np.ndarray, y: np.ndarray, n_iters: int = 1000, burn_in_perc: float = 0.5):
        '''
        TODO: Docs
        '''
        n, d = X.shape
        y_mat = y.reshape(-1, 1)
        
        # Initialize parameters
        beta = 1
        
        # Initialize result vectors
        self.weight_samples = np.zeros((n_iters, d))
        self.beta_samples = np.zeros(n_iters)
        self.log_posteriors = np.zeros(n_iters)
        
        # Perform Gibbs sampling
        for i in range(n_iters):
            
            # Sample parameters
            weights = self.__sample_weights__(X, y_mat, beta)
            beta = self.__sample_beta__(X, y_mat, weights)
            
            # Calculate unnormalized log-posterior
            lp = self.__calc_log_posterior__(X, y_mat, weights, beta)
            
            # Report back results
            self.weight_samples[i] = weights.reshape(d,)
            self.beta_samples[i] = beta
            self.log_posteriors[i] = lp
        
        # Find best parameters
        weight_hat = self.__get_map_params__(self.weight_samples)
        beta_hat = self.__get_map_params__(self.beta_samples)
        self.best_params = {
            'weights': weight_hat,
            'beta': beta_hat
        }
        
        self.ws = self.weight_samples

        # Assess performance
        lp_hat = self.__calc_log_posterior_data__(X, y_mat, weight_hat, beta_hat)
        bic = self.__bic__(lp_hat, n, d)
        aic = self.__aic__(lp_hat, n, d)
        self.performance = np.array([bic, aic])
        
        # Calculate burn-in
        self.burn_in = int(n_iters * burn_in_perc)
        return self
    
    def get_results(self):
        '''
        TODO: Docs
        '''
        
        # Create lists of samples and corresponding names
        param_samples = []
        param_names = []
        d = self.weight_samples.shape[1]
        
        # Add weight samples/names
        for i in range(d):
            param_samples.append(self.weight_samples[:, i])
            
            suffix = ''
            if i == 0:
                suffix = ' (intercept)'
            elif i == 1:
                suffix = ' (slope)'
            param_names.append(f'w{i}{suffix}')
        
        # Add beta samples/names
        param_samples.append(self.beta_samples)
        param_names.append('β')
        
        return param_samples, param_names