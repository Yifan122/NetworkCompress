blocktype = 'BasiBlock'
blocktype = 1 if blocktype == 'BasicBlock' else 2
b = 'conv{}'.format(blocktype)
print(b)