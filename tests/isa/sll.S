.globl _start
_start:
  addi t0, x0, 3
  slli t1, t0, 4
  srli t2, t1, 1
  addi t3, x0, -16
  srai t4, t3, 2
  xori t5, t2, 7
  ori t5, t5, 32
  andi t5, t5, 63
  add a0, t5, t4
  ebreak
