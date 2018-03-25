"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict(int,int)

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freqDict = {}
    for elem in text:
        if elem in freqDict:
            freqDict[elem] += 1
        else:
            freqDict[elem] = 1
    return freqDict


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

result1 = h.HuffmanNode(None, h.HuffmanNode(None, h.HuffmanNode('I', None, None), h.HuffmanNode('P', None, None)), h.HuffmanNode(None, h.HuffmanNode('E', None, None), h.HuffmanNode(None, h.HuffmanNode(None, h.HuffmanNode('T', None, None), h.HuffmanNode(None, h.HuffmanNode('G', None, None), h.HuffmanNode('S', None, None))), h.HuffmanNode('A', None, None))))


    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True

    SOURCES
    https://www.codecademy.com/en/forum_questions/548e5b1b9376762f5b007e5a
    """
    huffmanNodes = sortDict(freq_dict)
    while len(huffmanNodes) > 1:
        small1, small2 = huffmanNodes.pop(), huffmanNodes.pop()
        newTup = (small1[0] + small2[0], HuffmanNode(None, small1[1], small2[1]))
        huffmanNodes.append(newTup)
        #huffmanNodes = sorted(huffmanNodes, key=lambda hf: hf[0], reverse=True)
        huffmanNodes.sort()
        huffmanNodes = huffmanNodes[::-1]
    if len(huffmanNodes) > 0:
        return huffmanNodes[0][1]
    else:
        return HuffmanNode()

def sortDict(freqDict):
    """ Sorts the freqDict and returns a list of tuples (frequency, HuffmanNode(symbol))
    in descending order of frequency.

    SOURCES
    https://wiki.python.org/moin/HowTo/Sorting
    https://stackoverflow.com/questions/9460406/sort-dict-by-value-and-return-dict-not-list-of-tuples
    https://stackoverflow.com/questions/15712210/python-3-2-lambda-syntax-error?lq=1
    """
    sortedList = [(freq, HuffmanNode(symbol)) for symbol, freq in freqDict.items()]
    return sorted(sortedList, key=lambda kv:kv[0], reverse=True)

def get_codes(tree):

    """ Return a dict mapping symbols from Huffman tree to codes.
    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """

    codeDict = {}

    def _pathDict(root, bitPath = ""):
        nonlocal codeDict
        if not root:
            return {}
        elif root.symbol is None: # is a parent (no symbol), is None
            _pathDict(root.left, bitPath + "0")
            _pathDict(root.right, bitPath + "1")
        elif root.symbol is not None: # is a child (has a symbol) is not None
            if bitPath == "":
                bitPath = "1"
            codeDict[root.symbol] = bitPath
        return codeDict
    return _pathDict(tree)

def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)

[3, 2, None1, 9, 10, None2, Root] - Post Order - Left, Right, Root
[Root, None1, 3, 2, None2, 9, 10] - Pre order - Root, Left, Right
[3, None1, 2, Root, 9, None2, 10] - In Order - Left, Root, Right

    ROOT
    /  \
None1   None2
/ \    /  \
3  2   9   10

    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    num =-1
    def _postorder(tree):
        nonlocal num
        if not tree: # not a leaf
            return
        # recur left subtree
        # recur right subtree
        # check the nodes
        _postorder(tree.left)
        _postorder(tree.right)
        if tree.symbol is None:
            tree.number = 0
            num += 1
            tree.number += num
    _postorder(tree)

def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    codesDict = get_codes(tree) # {symbol: bits}
    total = 0
    totalFreq = 0
    bitLength = []

    for symbol, bits in codesDict.items():
        bitLength.append(len(codesDict[symbol]) * freq_dict[symbol])

    for freq in list(freq_dict.values()):
        totalFreq += freq

    totalSum = sum(bitLength)
    avg = totalSum/totalFreq
    return avg


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mapping from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    finalList = []
    byte = ""
    start = 0
    end = 8

    for number in list(text):
        byte += codes[number]
        #print(byte)
    while start < len(byte):
        finalList.append(byte[start: end])
        start = end
        end += 8
        #print(finalList)

    return bytes([bits_to_byte(c) for c in finalList])



def tree_to_bytes(tree):
    """ Return a bytes representation of the Huffman tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    finalList = b""
    
    if not tree.left.is_leaf():
        finalList += tree_to_bytes(tree.left)

    if not tree.right.is_leaf():
        finalList += tree_to_bytes(tree.right)

    if tree.left.is_leaf():
        finalList += bytes([0, tree.left.symbol])
    else:
        finalList += bytes([1, tree.left.number])

    if tree.right.is_leaf():
        finalList += bytes([0, tree.right.symbol])
    else:
        finalList += bytes([1, tree.right.number])
        
    return finalList

    #def _postorder(tree):
    #    nonlocal finalList
    #    if not tree: # not a leaf
    #        return
        # recur left subtree
        # recur right subtree
        # check the nodes
    #    _postorder(tree.left)
    #    _postorder(tree.right)
        # Is it a leaf?
    #    if tree.left is None and tree.right is None:
    #        finalList += bytes([0,tree.symbol])
    #    elif tree.number is not None:
    #        finalList += bytes([1,tree.number])

    #_postorder(tree)
    #return finalList[:-2]
    
def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file to compress
    @param str out_file: output file to store compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in node_lst.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)), HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))

       Root
     /      \
    None1   None2
    /  \    /    \
   10  12   5    7

    """
    root = node_lst[root_index]
    leftNode = HuffmanNode(root.l_data)
    rightNode = HuffmanNode(root.r_data)
    if root.l_type == 1:
        # RECUR FOR THE LEFT SUBTREE IF INTERNAL NODE
        leftNode = generate_tree_general(node_lst, root.l_data)
    if root.r_type == 1:
        # RECUR FOR THE RIGHT SUBTREE IF INTERNAL NODE
        rightNode = generate_tree_general(node_lst, root.r_data)
    return HuffmanNode(None, leftNode, rightNode)

def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that node_lst represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)), HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))

          Root
        /       \
    None1       None2
    /    \      /    \
    5    7      10    12
    """
    # todo


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: number of bytes to decompress from text.
    @rtype: bytes
    """

    # convert get_codes(tree) so that its values (codes) are mapped to keys (symbols)
    swapDict = {}
    codesDict = get_codes(tree)
    for symbol, bit in codesDict.items():
        swapDict[bit] = symbol

    i = 0
    myStr = ""
    finalList = []

    # 1. for loop on the range of size:
    for a in range(size):

        for b in swapDict:

            if len(b) > len(myStr) and i < len(text):
                #2. for every bit in text: use 'bytes to bits' on that index of text
                myStr += byte_to_bits(text[i])
                #3. add this conversion to an emptry string "myStr" and increment the index by 1
                i += 1
            #4 check if myStr[:len(b)] is the current key in swapDict
            if myStr[:len(b)] == b:
                finalList.append(swapDict[b])

                myStr = myStr[len(b):]

                break
    return bytes(finalList)

def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    sortedFreqList = sortDict(freq_dict)
    sortedFreqList = [(i[0], i[1].symbol) for i in sortedFreqList]
    sortedFreqList.sort()
    def _inOrder(tree, sortedFreqList):
        if not tree:
            return
        # recur left Subtree
        # check root
        # recur right Subtree
        _inOrder(tree.left, sortedFreqList)
        if tree.symbol is not None:
            tree.symbol = sortedFreqList.pop()[1]
        _inOrder(tree.right, sortedFreqList)
    _inOrder(tree, sortedFreqList)

if __name__ == "__main__":
    # TODO: Uncomment these when you have implemented all the functions
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
