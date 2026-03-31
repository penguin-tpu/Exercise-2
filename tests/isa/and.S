.globl _start
_start:
  addi t0, x0, 3
  addi t1, x0, 4
  sll t2, t0, t1
  addi t3, x0, -64
  sra t4, t3, t0
  addi t5, x0, 64
  srl t6, t5, t1
  xor a0, t2, t6
  and a0, a0, t2
  add a0, a0, t4
  ebreak
