.section .data
INPUT:
  .word 2, 4, 6, 8

.section .text
.globl _start
_start:
  la s0, INPUT
  lui s1, 0x20000
  addi s2, x0, 4
  addi t0, x0, 0

loop:
  beq s2, x0, done
  lw t1, 0(s0)
  add t0, t0, t1
  addi s0, s0, 4
  addi s2, s2, -1
  jal x0, loop

done:
  sw t0, 0(s1)
  addi a0, t0, 0
  ebreak
