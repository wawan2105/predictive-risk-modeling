import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.preprocessing import load_data,split_data,pipeline
from src.evaluate import evaluate,threshold
class IntegrityTests(unittest.TestCase):
    def test_disjoint_reproducible_splits(self):
        a,b,c,ya,yb,yc = split_data()
        groups = [set(x.index) for x in [a,b,c]]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        self.assertEqual(len(set.union(*groups)),len(load_data()[0]))
        self.assertTrue(a.equals(split_data()[0]))
        self.assertEqual(int(load_data()[1].sum()),212)
    def test_missing_value_inference(self):
        a,b,c,ya,yb,yc = split_data()
        model = pipeline(LogisticRegression(max_iter=3000)).fit(a,ya)
        b = b.copy()
        b.iloc[0,0] = np.nan
        self.assertTrue(np.isfinite(model.predict_proba(b)).all())
    def test_evaluation(self):
        y=np.array([0,0,1,1]); p=np.array([.1,.2,.8,.9])
        m=evaluate(y,p)
        self.assertEqual(m['f1'],1)
        self.assertEqual((m['tn'],m['fp'],m['fn'],m['tp']),(2,0,0,2))
        self.assertGreaterEqual(threshold(y,p),.05)
if __name__ == '__main__':
    unittest.main()
