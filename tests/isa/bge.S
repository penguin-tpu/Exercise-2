.globl _start
_start:
  addi t0, x0, 4
  addi t1, x0, 4
  bge t0, t1, equal_ok
  addi a0, x0, 1
  ebreak
equal_ok:
  addi t2, x0, -1
  bge t1, t2, done
  addi a0, x0, 2
  ebreak
done:
  addi a0, x0, 14
  ebreak
