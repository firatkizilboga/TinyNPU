.section .text.start
.global _start
.type _start, @function

.equ STDOUT_REG, 0x10000000
.equ EXIT_REG,   0x20000004

.extern __ieee754_sqrtf

_start:
    /* progress */
    li t0, STDOUT_REG
    li t1, 'S'
    sw t1, 0(t0)

    /* initialize global pointer */
    .option push
    .option norelax
1:  auipc gp, %pcrel_hi(__global_pointer$)
    addi  gp, gp, %pcrel_lo(1b)
    .option pop

    /* initialize stack pointer */
    la sp, _sp

    /* enable floating-point state */
    li t0, (3 << 13)
    csrs mstatus, t0

    /* call __ieee754_sqrtf(2.0f) */
    li t0, 0x40000000
    fmv.w.x fa0, t0
    call __ieee754_sqrtf
    fmv.x.w t1, fa0

    li t2, 0x3fb504f3
    li t3, EXIT_REG
    beq t1, t2, 1f

    li t0, STDOUT_REG
    li t4, 'F'
    sw t4, 0(t0)
    li t4, 1
    sw t4, 0(t3)
    wfi

1:
    li t0, STDOUT_REG
    li t4, 'P'
    sw t4, 0(t0)
    sw zero, 0(t3)
    wfi

.size _start, .-_start
