.section .data
A:
  .word 1, 2, 3, 4
B:
  .word 10, 20, 30, 40

.section .text
.globl _start
_start:
  la s0, A
  la s1, B
  lui s2, 0x20000
  addi s3, x0, 4

loop:
  beq s3, x0, done
  lw t0, 0(s0)
  lw t1, 0(s1)
  add t2, t0, t1
  sw t2, 0(s2)
  addi s0, s0, 4
  addi s1, s1, 4
  addi s2, s2, 4
  addi s3, s3, -1
  jal x0, loop

done:
  lui t3, 0x20000
  lw a0, 12(t3)
  ebreak
