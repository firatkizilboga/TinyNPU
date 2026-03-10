class MemoryManager:
    """
    Manages the Unified Buffer memory space.
    """
    def __init__(self, start_addr=0):
        self.next_free_addr = start_addr
        self.symbol_to_addr = {}
        self.symbol_to_size = {}

    def allocate(self, name, word_count):
        """
        Allocates a contiguous block of UB words for a symbol.
        Returns the starting address.
        """
        if name in self.symbol_to_addr:
            # Check if current allocation is large enough? 
            # For now, as per user rule: declare_data means new space.
            # But if it's the SAME name, we might want to reuse or warn.
            return self.symbol_to_addr[name]

        addr = self.next_free_addr
        self.symbol_to_addr[name] = addr
        self.symbol_to_size[name] = word_count
        self.next_free_addr += word_count
        return addr

    def get_addr(self, name):
        if name not in self.symbol_to_addr:
            raise KeyError(f"Symbol '{name}' has not been allocated.")
        return self.symbol_to_addr[name]
