from subprocess import Popen, PIPE
proc = Popen(". ~/anaconda3/etc/profile.d/conda.sh && conda activate sketch && ./test.sh",
      shell=True,
      executable="/bin/bash",
      stdin=PIPE,
      stdout=PIPE,
      stderr=PIPE
     )
print("STDOUT:\n")
print(proc.stdout.read().decode('utf8'))
print("STDERR:\n", proc.stderr.read().decode('utf8'))