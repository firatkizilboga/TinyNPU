import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import random

# Constants from defines.sv
DATA_WIDTH = 16
ARRAY_SIZE = 4
BUFFER_WIDTH = DATA_WIDTH * ARRAY_SIZE  # 64 bits
BUFFER_DEPTH = 1024


def pack_row(values):
    """Pack a list of values into a single BUFFER_WIDTH integer.
    
    values[0] goes to LSB, values[N-1] goes to MSB.
    """
    packed = 0
    for i, val in enumerate(values):
        packed |= (val & ((1 << DATA_WIDTH) - 1)) << (i * DATA_WIDTH)
    return packed


def unpack_row(packed):
    """Unpack a BUFFER_WIDTH integer into a list of values."""
    mask = (1 << DATA_WIDTH) - 1
    values = []
    for i in range(ARRAY_SIZE):
        values.append((packed >> (i * DATA_WIDTH)) & mask)
    return values


@cocotb.test()
async def test_write_read_basic(dut):
    """Test basic write and read operations."""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.wr_en.value = 0
    dut.wr_addr.value = 0
    dut.wr_data.value = 0
    dut.input_addr.value = 0
    dut.weight_addr.value = 0
    
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    dut._log.info("Testing basic write/read...")
    
    # Write test data to address 0
    test_row_0 = [0x1111, 0x2222, 0x3333, 0x4444]
    packed_0 = pack_row(test_row_0)
    
    dut.wr_en.value = 1
    dut.wr_addr.value = 0
    dut.wr_data.value = packed_0
    await RisingEdge(dut.clk)
    
    # Write test data to address 1
    test_row_1 = [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD]
    packed_1 = pack_row(test_row_1)
    
    dut.wr_addr.value = 1
    dut.wr_data.value = packed_1
    await RisingEdge(dut.clk)
    
    # Disable write
    dut.wr_en.value = 0
    
    # Read from address 0 (inputs) and address 1 (weights)
    dut.input_addr.value = 0
    dut.weight_addr.value = 1
    await RisingEdge(dut.clk)  # Wait for read to register
    await RisingEdge(dut.clk)  # Output appears after 1 cycle
    
    # Check results
    input_read = dut.input_data.value.integer
    weight_read = dut.weight_data.value.integer
    
    input_unpacked = unpack_row(input_read)
    weight_unpacked = unpack_row(weight_read)
    
    dut._log.info(f"Input read:  {[hex(v) for v in input_unpacked]}")
    dut._log.info(f"Weight read: {[hex(v) for v in weight_unpacked]}")
    
    assert input_unpacked == test_row_0, f"Input mismatch: expected {test_row_0}, got {input_unpacked}"
    assert weight_unpacked == test_row_1, f"Weight mismatch: expected {test_row_1}, got {weight_unpacked}"
    
    dut._log.info("✅ Basic write/read test PASSED!")


@cocotb.test()
async def test_independent_read_ports(dut):
    """Test that input and weight ports can read different addresses simultaneously."""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.wr_en.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    dut._log.info("Testing independent read ports...")
    
    # Write to multiple addresses
    test_data = {}
    for addr in range(8):
        row = [addr * 4 + i for i in range(ARRAY_SIZE)]
        test_data[addr] = row
        
        dut.wr_en.value = 1
        dut.wr_addr.value = addr
        dut.wr_data.value = pack_row(row)
        await RisingEdge(dut.clk)
    
    dut.wr_en.value = 0
    
    # Test reading different addresses on both ports
    for input_addr in range(0, 8, 2):
        weight_addr = input_addr + 1
        
        dut.input_addr.value = input_addr
        dut.weight_addr.value = weight_addr
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        
        input_read = unpack_row(dut.input_data.value.integer)
        weight_read = unpack_row(dut.weight_data.value.integer)
        
        assert input_read == test_data[input_addr], \
            f"Input port mismatch at addr {input_addr}"
        assert weight_read == test_data[weight_addr], \
            f"Weight port mismatch at addr {weight_addr}"
    
    dut._log.info("✅ Independent read ports test PASSED!")


@cocotb.test()
async def test_random_access(dut):
    """Test random write/read patterns."""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.wr_en.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    dut._log.info("Testing random access patterns...")
    
    # Generate random test data
    random.seed(42)
    num_entries = 32
    test_data = {}
    
    for _ in range(num_entries):
        addr = random.randint(0, BUFFER_DEPTH - 1)
        row = [random.randint(0, 0xFFFF) for _ in range(ARRAY_SIZE)]
        test_data[addr] = row
        
        dut.wr_en.value = 1
        dut.wr_addr.value = addr
        dut.wr_data.value = pack_row(row)
        await RisingEdge(dut.clk)
    
    dut.wr_en.value = 0
    
    # Verify all written data
    errors = 0
    for addr, expected in test_data.items():
        dut.input_addr.value = addr
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        
        actual = unpack_row(dut.input_data.value.integer)
        if actual != expected:
            dut._log.error(f"Mismatch at addr {addr}: expected {expected}, got {actual}")
            errors += 1
    
    assert errors == 0, f"Random access test failed with {errors} errors"
    dut._log.info(f"✅ Random access test PASSED! ({len(test_data)} entries verified)")


@cocotb.test()
async def test_reset_clears_outputs(dut):
    """Test that reset clears the output registers."""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset and write some data
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # Write non-zero data
    dut.wr_en.value = 1
    dut.wr_addr.value = 0
    dut.wr_data.value = 0xFFFFFFFFFFFFFFFF
    await RisingEdge(dut.clk)
    dut.wr_en.value = 0
    
    # Read it back
    dut.input_addr.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    
    # Verify data is there
    assert dut.input_data.value.integer == 0xFFFFFFFFFFFFFFFF, "Data not written"
    
    # Apply reset
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    
    # Check outputs are cleared
    assert dut.input_data.value.integer == 0, "Reset didn't clear input_data"
    assert dut.weight_data.value.integer == 0, "Reset didn't clear weight_data"
    
    dut._log.info("✅ Reset clears outputs test PASSED!")
