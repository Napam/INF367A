from EM_GMM import GMM
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
np.random.seed(42069)

if __name__ == '__main__':
    def task1():
        df = pd.read_csv('largest_cities.csv')
        names = df['city']
        X = df[['lng', 'lat']].values

        bics2d = []
        models2d = []

        for i in range(6):
            models = [GMM(k=i).fit(X) for i in range(2, 19)]
            models2d.append(models)
            bics2d.append([m.get_bic() for m in models])

        bics2d = np.array(bics2d)
        models2d = np.array(models2d)
        best_idx = np.unravel_index(np.argmax(bics2d), bics2d.shape)
        best_model = models2d[best_idx]

        print(bics2d)
        print(best_idx)

        bics = bics2d[best_idx[0]]
        models = models2d[best_idx[0]]
        print(f'''
            Best model:
            Number of components = {best_model.k}
            Final loglikelihood = {best_model.hood_history[-1]}
            BIC = {bics[best_idx[1]]}
        ''')

        dicto = {'models':models}
        for i, b in enumerate(bics2d, start=1):
            dicto[f'BIC_{i}']=b

        df_results = pd.DataFrame(dicto)

        print(df_results.to_latex(index=False, float_format=lambda x: f'{x:.2f}'))

        best_model.plot_result(show=False, figsize=(9,5))
        # plt.savefig('GMM_task1.pdf')
        best_model.show()
    
    def task2():
        df = pd.read_csv('ex8_2.csv', header=None)
        X = df.values.flatten()
        
        # Inherit old GMM class for its plotting abilities
        class GMM2(GMM):
            def _prepare_before_fit(self, X: np.ndarray) -> None:
                '''
                Prepares object attributes and such before fitting
                '''
                super()._prepare_before_fit(X)
                # Set component 0 mean to 0 
                self.components[0]['mean'] = 0
                # Set standard deviations to 1
                self.components['cov'].fill(1)

            def _M_step(self) -> None:
                w = self.weights[:,1]
                w_sum = w.sum()      
                # Update component mean
                self.components[1]['mean'] = w@self.X/w_sum
                # Update component mixing probability
                self.components[1]['mix'] = w_sum/self.N
                self.components[0]['mix'] = 1 - self.components[1]['mix']

        # Override class methods
        gmm = GMM2(k=2)
        gmm.fit(X.reshape(-1,1))
        # gmm.fit_animate(X.reshape(-1,1))
        gmm.plot_result(show=False, figsize=(9,5))
        plt.savefig('GMM_task2.pdf')
        gmm.show()

    task2()

    