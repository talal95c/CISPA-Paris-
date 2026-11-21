import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from decision import (SAME, VARIANTS, cluster, decide, distance, mcc, parse_burst,
                      read_answers, score, signed_projection, split_items,
                      standardized_u, u_items, write_submission)
from phase0 import draw, load, schedule


class DecisionTests(unittest.TestCase):
    def test_parser_normal_empty_malformed_duplicate(self):
        book={'q7k':dict(options=['Yes','No']),'a2b':dict(options=['A','B'])}
        self.assertEqual(parse_burst('q7k Yes\na2b B',book),{'q7k':0,'a2b':1})
        for text in ['', '[q7k]: Yes', 'q7k Yes because...', 'q7k Yes\nq7k No']:
            self.assertEqual(parse_burst(text,book)['q7k'],10)
        with self.assertRaises(ValueError):
            parse_burst(None,book)

    def test_jsonl_ingestion(self):
        manifest={'0':dict(half=0,items=[dict(item=0,code='q7k',options=['Yes','No'])])}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'answers.jsonl'
            row=json.dumps(dict(endpoint='ep-01',id='0',repeat=0,response='q7k Yes'))+'\n'
            path.write_text(row)
            counts=read_answers(path,manifest)['ep-01']
            self.assertEqual(counts[0,0,0,0],1)
            path.write_text(row+row)
            with self.assertRaises(ValueError):
                read_answers(path,manifest)

    def test_u_exact_unbiased_binomial_enumeration(self):
        # Enumerate the sampling distribution, not a tolerance-based Monte Carlo test.
        import math
        p,q,n,m=.3,.7,3,4
        expected=0.
        for a,b in itertools.product(range(n+1),range(m+1)):
            weight=math.comb(n,a)*p**a*(1-p)**(n-a)*math.comb(m,b)*q**b*(1-q)**(m-b)
            expected+=weight*distance(np.array([[a,n-a]]),np.array([[b,m-b]]))
        self.assertAlmostEqual(expected,2*(p-q)**2)
        self.assertTrue(np.isnan(u_items([[1,0]],[[0,1]])[0]))
        self.assertEqual(distance(np.zeros((0,2)),np.zeros((0,2))),0)
        with self.assertRaises(ValueError):
            standardized_u([[2,0]],[[0,2]],0,0)

    def test_schedule_disjoint_bursts_and_budget(self):
        _,items=load('135m')
        splits=split_items(items)
        self.assertFalse(set(splits['tune'])&set(splits['holdout']))
        self.assertFalse(set(splits['tune'])&set(splits['type']))
        self.assertEqual(len(splits['tune']),125)
        ids,probes,n=schedule(splits['tune'],60,np.random.default_rng(0))
        self.assertEqual(n.sum(),1364)
        self.assertTrue((n.sum((1,2))==682).all())
        self.assertTrue((n[:,:,0]>=2).all())
        self.assertEqual(len(probes),22)
        p=np.zeros((1,60,2,11));p[:,:,:,0]=1
        sample=draw(p,n,np.random.default_rng(1))
        np.testing.assert_array_equal(sample.sum(-1)[0],n)

    def test_score_matches_official_practice_scorer(self):
        # Official refs used ONLY here, never by calibration/evaluation.
        import csv
        with Path('/home/ubuntu/me/practice_answers.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        for merge in [False,True]:
            families=[x['family'] for x in rows]
            truth=[x['model'] for x in rows]
            labels=['S']*len(rows) if merge else truth
            result=score(families,truth,families,labels)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'submission.csv'
                write_submission(path,[x['endpoint'] for x in rows],families,labels)
                output=subprocess.check_output(['/home/ubuntu/me/venv/bin/python','/home/ubuntu/me/score_practice.py',str(path)],text=True)
                self.assertIn(f"): {result['model']:.3f}",output)
        self.assertEqual(mcc([],[]),0)
        self.assertEqual(score([0,0,0,1,1],[0,0,1,2,3],[0,0,0,1,1],[0,0,1,2,3])['score'],1)

    def test_complete_linkage_and_crossfit(self):
        matrix=np.array([[0,.1,.8],[.1,0,.2],[.8,.2,0]])
        labels=cluster(matrix,.3)
        self.assertEqual(labels[0],labels[1]);self.assertNotEqual(labels[0],labels[2])
        counts=np.zeros((3,2,1,2),int)
        counts[0,:,:,0]=10;counts[1,:,:,1]=10;counts[2,:,:,0]=5;counts[2,:,:,1]=5
        projection=signed_projection(counts,[0],[1])
        self.assertLess(projection[0],projection[2]);self.assertLess(projection[2],projection[1])

    def test_R_C_pool_and_singletons(self):
        counts=np.zeros((4,2,1,11),int)
        counts[:,:,:,0]=100
        counts[2,:,:,0]=0;counts[2,:,:,1]=100
        counts[3,:,:,0]=95;counts[3,:,:,1]=5
        params=dict(null_mean=0,null_sd=1,family_threshold=3,pool_threshold=.1,
                    enable_watermark=False,enable_temperature=False,enable_ranking=True,
                    enable_axis=False,rank_stat='distance',rank_precision2=.8,rank_precision3=.8)
        codes=np.zeros((4,1,2,11))
        _,r=decide(counts,codes,[True,False,False,False],params,'R')
        _,c=decide(counts,codes,[True,False,False,False],params,'C')
        self.assertEqual(r[0],r[1]);self.assertEqual(r[0],r[3]);self.assertNotEqual(r[0],r[2])
        self.assertEqual(c[0],c[1]);self.assertNotEqual(c[0],c[3])
        _,none=decide(counts,codes,[False]*4,params,'R')
        self.assertEqual(len(set(none)),4)


if __name__=='__main__':
    unittest.main()
