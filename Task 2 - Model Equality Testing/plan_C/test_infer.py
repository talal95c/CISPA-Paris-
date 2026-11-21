import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class InferenceIntegrationTests(unittest.TestCase):
    def test_jsonl_to_submission_and_empty_error(self):
        root=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            manifest={str(i):dict(half=i%2,items=[dict(item=0,code='q7k',options=['Yes','No'])]) for i in range(8)}
            (directory/'manifest.json').write_text(json.dumps(manifest))
            rows=[dict(endpoint=f'ep-{e:02}',id=str(i),repeat=0,response='q7k '+('Yes' if e<3 else 'No')) for e in [1,2,3] for i in range(8)]
            (directory/'answers.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            (directory/'flags.json').write_text(json.dumps({'ep-01':True}))
            command=[sys.executable,str(root/'infer.py'),'--answers',str(directory/'answers.jsonl'),'--manifest',str(directory/'manifest.json'),'--flags',str(directory/'flags.json'),'--output',str(directory/'submission.csv')]
            output=subprocess.check_output(command,text=True)
            self.assertIn('Wrote 3 endpoints',output)
            lines=(directory/'submission.csv').read_text().splitlines()
            self.assertEqual(lines[0],'endpoint,family,model')
            self.assertEqual(len(lines),4)
            (directory/'answers.jsonl').write_text('')
            result=subprocess.run(command,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('No responses found',result.stderr)


if __name__=='__main__':
    unittest.main()
