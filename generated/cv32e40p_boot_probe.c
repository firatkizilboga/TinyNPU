#include <stdio.h>
#include <stdlib.h>

volatile unsigned int runtime_cycle_start __attribute__((section(".noinit")));
volatile unsigned int runtime_cycle_post_bss __attribute__((section(".noinit")));
volatile unsigned int runtime_cycle_post_init __attribute__((section(".noinit")));
volatile unsigned int runtime_cycle_pre_main __attribute__((section(".noinit")));

int main(void)
{
    printf("boot_probe\n");
    printf("EXIT SUCCESS\n");
    return EXIT_SUCCESS;
}
