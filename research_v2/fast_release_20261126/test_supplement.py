import unittest
import pandas as pd
from quality_supplement import correlations
class SupplementTests(unittest.TestCase):
    def frame(self,constant=False):
        return pd.DataFrame([dict(scope='available',width=10,label=label,window_id=str(i),feature_id=f,value=(1 if constant else (i if f=='P01' else 3-i))) for label in (0,1) for i in range(4) for f in ('P01','P02')])
    def test_group_separation(self):
        r=correlations(self.frame());self.assertEqual(len(r),2)
        self.assertTrue(r.paired_n.eq(4).all());self.assertTrue(r.pearson.eq(-1).all())
    def test_constant_missing(self):
        r=correlations(self.frame(True));self.assertTrue(r.pearson.isna().all())
    def test_missing_pair_counts(self):
        d=self.frame();d.loc[d.window_id.eq('0'),'value']=float('nan')
        self.assertTrue(correlations(d).paired_n.eq(3).all())
if __name__=='__main__':unittest.main()
