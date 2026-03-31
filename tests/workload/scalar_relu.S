.section .data
INPUT:
  .word -3, 0, 5, -1

.section .text
.globl _start
_start:
  la s0, INPUT
  lui s1, 0x20000
  addi s2, x0, 4

loop:
  beq s2, x0, done
  lw t0, 0(s0)
  blt t0, x0, clamp_zero
  addi t1, t0, 0
  jal x0, store_value

clamp_zero:
  addi t1, x0, 0

store_value:
  sw t1, 0(s1)
  addi s0, s0, 4
  addi s1, s1, 4
  addi s2, s2, -1
  jal x0, loop

done:
  lui t2, 0x20000
  lw a0, 12(t2)
  ebreak
