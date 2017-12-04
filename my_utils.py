def write_list(l):
	with open('list.txt','wt') as tar:
		for e in l:
			tar.write(str(e) + '\n')