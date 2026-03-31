.globl _start
_start:
  addi a0, x0, 8
  jal ra, step_one
  addi a0, x0, 1
  ebreak
step_one:
  auipc t3, 0
  addi t3, t3, 16
  jalr x0, 0(t3)
  addi a0, x0, 2
target:
  addi a0, a0, 7
  ebreak
