import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import collect


class Reply:
    def __init__(self,status,body,headers=None):
        self.status_code=status;self.body=body;self.headers=headers or {};self.text=json.dumps(body)
    def json(self):
        return self.body


class CollectTests(unittest.TestCase):
    def test_protocol_and_unauthorized_live(self):
        prompts,manifest=collect.make_protocol()
        self.assertEqual(len(prompts),100)
        self.assertEqual([sum(x['kind']==k for x in prompts) for k in ['flag','long','burst','reserve']],[4,30,62,4])
        for i in [94,95]:
            self.assertEqual(prompts[i]['prompt'],prompts[i-60]['prompt'])
            self.assertEqual(manifest[str(i)]['items'],manifest[str(i-60)]['items'])
        self.assertTrue(all(len(p['items'])==22 for p in manifest.values() if p['kind']=='burst'))
        with tempfile.TemporaryDirectory() as directory, patch('socket.socket',side_effect=AssertionError('Network forbidden')):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                collect.main(['--live','--out',directory])
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                collect.main(['--live','--i-understand-budget','--out',directory])

    def test_transport_stub_retry_budget_journal_and_resume(self):
        prompts=[dict(id=str(i),prompt='test') for i in range(6)]
        calls=[];lock=threading.Lock();state=dict(used=0,active=0,max_active=0,busy=True)
        def transport(method,url,**kwargs):
            with lock:
                calls.append(method)
                if method=='GET':
                    return Reply(200,{'data':[{'id':'ref-01','budget':{'limit':100,'used':state['used']}}]})
                if state['busy']:
                    state['busy']=False
                    return Reply(429,{'detail':'busy'},{'Retry-After':'0'})
                state['active']+=1;state['max_active']=max(state['max_active'],state['active'])
                state['used']+=1
                body={'choices':[{'message':{'content':'q7k Yes'}}],'budget':{'limit':100,'used':state['used']},'extra':{'keep':'full JSON'}}
                state['active']-=1
                return Reply(200,body)
        with tempfile.TemporaryDirectory() as directory,patch('socket.socket',side_effect=AssertionError('Network forbidden')):
            directory=Path(directory)
            waits=[]
            collect.live(directory,prompts,['ref-01'],'http://never-called','test-key',transport=transport,wait=waits.append)
            self.assertEqual(waits,[0.])
            self.assertEqual(len(collect.read_jsonl(directory/'answers.jsonl')),6)
            success=[x for x in collect.read_jsonl(directory/'responses.jsonl') if x['event']=='success']
            self.assertTrue(all(x['response']['extra']['keep']=='full JSON' for x in success))
            before=len(calls)
            collect.live(directory,prompts,['ref-01'],'http://never-called','test-key',transport=transport)
            self.assertEqual(len(calls),before)
            self.assertLessEqual(state['max_active'],4)
            self.assertEqual(collect.PARALLEL,4)

    def test_budget_error_and_uncertain_request_are_not_retried(self):
        prompts=[dict(id='0',prompt='test')]
        def insufficient(method,url,**kwargs):
            self.assertEqual(method,'GET')
            return Reply(200,{'data':[{'id':'ep-01','budget':{'limit':100,'used':100}}]})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'Insufficient'):
                collect.live(Path(directory),prompts,['ep-01'],'http://never-called','test',transport=insufficient)
        def uncertain(method,url,**kwargs):
            if method=='GET':
                return Reply(200,{'data':[{'id':'ep-01','budget':{'limit':100,'used':0}}]})
            raise TimeoutError('uncertain delivery')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TimeoutError):
                collect.live(Path(directory),prompts,['ep-01'],'http://never-called','test',transport=uncertain)
            with self.assertRaisesRegex(ValueError,'Uncertain'):
                collect.recover(Path(directory))
        self.assertEqual(collect.retry_after('2.5'),2.5)
        with self.assertRaises(ValueError):
            collect.retry_after('99999')

    def test_recovery_and_protocol_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            row=dict(endpoint='ref-01',id='0',repeat=0,prompt='test',response='test')
            collect.append_json(directory/'responses.jsonl',dict(event='success',answer=row))
            self.assertEqual(collect.recover(directory),{('ref-01','0',0)})
            self.assertEqual(len(collect.read_jsonl(directory/'answers.jsonl')),1)
            collect.save_fixed(directory/'plan.json','one')
            with self.assertRaises(ValueError):
                collect.save_fixed(directory/'plan.json','two')

    def test_mock_resume_repairs_corrupt_answers_without_recollecting(self):
        prompts=[dict(id=str(i),prompt='test') for i in range(2)]
        manifest={p['id']:dict(items=[]) for p in prompts}
        paid=[dict(endpoint=f'ref-{i:02}',id='0',repeat=0,prompt='test',response=f'paid {i}') for i in range(1,4)]
        with tempfile.TemporaryDirectory() as directory,patch('socket.socket',side_effect=AssertionError('Network forbidden')):
            directory=Path(directory)
            answers=directory/'answers.jsonl';journal=directory/'responses.jsonl'
            answers.write_bytes((json.dumps(paid[0])+'\ncorrupt middle line\n{"endpoint":').encode())
            events=[dict(event='success',answer=row) for row in paid[1:]]
            journal.write_bytes(('\n'.join(json.dumps(event) for event in events)+'\n').encode())
            warnings=io.StringIO()
            with patch.object(collect,'make_protocol',return_value=(prompts,manifest)),patch.object(collect,'CACHE') as cache,patch.object(collect.np,'load',return_value=contextlib.nullcontext({v:[] for v in collect.VARIANTS})),patch.object(collect,'append_json',wraps=collect.append_json) as append,contextlib.redirect_stderr(warnings),contextlib.redirect_stdout(io.StringIO()):
                cache.glob.return_value=iter([Path('unused.npz')])
                collect.main(['--mock','--target','refs','--out',str(directory)])
            self.assertEqual(warnings.getvalue().count('WARNING: ignoring corrupt JSONL line'),2)
            new_keys={collect.answer_key(call.args[1]['answer']) for call in append.call_args_list if call.args[0]==journal}
            paid_keys={collect.answer_key(row) for row in paid}
            expected={(f'ref-{i:02}',str(j),0) for i in range(1,10) for j in range(2)}
            self.assertEqual(new_keys,expected-paid_keys)
            for path in [answers,journal]:
                rows=[json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(rows),18 if path==answers else 17)
                self.assertTrue(path.read_bytes().endswith(b'\n'))
            rows=collect.read_jsonl(answers)
            self.assertEqual({collect.answer_key(row) for row in rows},expected)
            for row in paid:
                self.assertIn(row,rows)
            self.assertEqual(collect.recover(directory),expected)

    def test_corrupt_event_journal_refuses_resume_without_changes(self):
        success=b'{"event":"success","answer":{"endpoint":"ref-01","id":"0"}}\n'
        corrupt=[success+b'{"event":"request","key":',
                 success+b'corrupt middle line\n'+success,
                 success+b'\xc3',
                 success+b'{"event":"request","key":["ref-01","1",0]}']
        for original in corrupt:
            with self.subTest(original=original),tempfile.TemporaryDirectory() as directory:
                directory=Path(directory)
                journal=directory/'responses.jsonl';answers=directory/'answers.jsonl'
                journal.write_bytes(original)
                answers.write_bytes(b'{"endpoint":')
                with patch.object(collect,'append_json') as append:
                    with self.assertRaisesRegex(ValueError,'manual inspection required'):
                        collect.recover(directory)
                append.assert_not_called()
                self.assertEqual(journal.read_bytes(),original)
                self.assertEqual(answers.read_bytes(),b'{"endpoint":')
                self.assertEqual(set(directory.iterdir()),{journal,answers})

    def test_jsonl_empty_missing_and_valid_unterminated_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'answers.jsonl'
            self.assertEqual(collect.read_jsonl(path,repair_corrupt=True),[])
            path.touch()
            self.assertEqual(collect.read_jsonl(path,repair_corrupt=True),[])
            path.write_text('{"id": 0}')
            self.assertEqual(collect.read_jsonl(path,repair_corrupt=True),[{'id':0}])
            collect.append_json(path,{'id':1})
            self.assertEqual([json.loads(line) for line in path.read_text().splitlines()],[{'id':0},{'id':1}])

    def test_jsonl_failed_repair_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'answers.jsonl'
            original=b'{"id":0}\n{"id":'
            path.write_bytes(original)
            with patch.object(collect.os,'replace',side_effect=OSError('repair failed')),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(OSError,'repair failed'):
                    collect.read_jsonl(path,repair_corrupt=True)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(list(path.parent.iterdir()),[path])


if __name__=='__main__':
    unittest.main()
