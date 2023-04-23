# code to read /proc/meminfo and get memory usage
import os
import sys
import time

def get_meminfo():
	meminfo = {}
	with open('/proc/meminfo') as f:
		for line in f:
			parts = line.split(':')
			if len(parts) != 2:
				continue
			key = parts[0].strip()
			val = parts[1].strip()
			meminfo[key] = val
	return meminfo


def get_mem_usage():
	meminfo = get_meminfo()
	mem_total = int(meminfo['MemTotal'].split()[0])
	mem_avail = int(meminfo['MemAvailable'].split()[0])
	swap_total = int(meminfo['SwapTotal'].split()[0])
	swap_free = int(meminfo['SwapFree'].split()[0])
	mem_used = mem_total - mem_avail
	swap_used = swap_total - swap_free
	return mem_used,swap_used

# code to read /proc/meminfo and get swap usage

if __name__ == "__main__":
	for i in range(0,100):
		print(get_mem_usage())
		time.sleep(0.1)