#!/usr/bin/env python
#Author: Vojtech Kulvait
#Description: GPU management through Redis server BRPOP and LPUSH
#Date: 2024
#License: GPL-3.0

import pycuda
import redis
import argparse
import datetime
import os
from termcolor import colored
from pycuda import compiler
import pycuda.driver as drv
import pyopencl as cl



def checkPID(PID):
	try:
		os.kill(PID, 0)
	except OSError:
		return False
	else:
		return True

def initRedisObjects(rs, prefix, managedList, GPUcount, force=False):
	lastinit_data = rs.hgetall("%s_INIT"%(prefix))
	if len(lastinit_data.keys()) == 0 or force:
		rs.delete("%s_IDLE"%(prefix))
		rs.delete("%s_MANAGED"%(prefix))
		rs.delete("%s_EVENTS"%(prefix))
		rs.delete("%s_INIT"%(prefix))
		for i in managedList:
			rs.lpush("%s_IDLE"%(prefix), i)
			rs.lpush("%s_MANAGED"%(prefix), i)
		rs.hset("%s_INIT"%(prefix), "TIME", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		rs.hset("%s_INIT"%(prefix), "GPUMANAGED", ",".join([str(i) for i in managedList]))
		rs.hset("%s_INIT"%(prefix), "GPUCOUNT", GPUcount)
		rs.lpush("%s_EVENTS"%(prefix), "%s: Redis initialized, %d managed GPUs."%(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(managedList)))
		return 0
	else:
		print("Objects already initialized, add force to proceed.")
		return -1


def getManagedGPUs(rs, prefix):
	managed = rs.lrange("%s_MANAGED"%(prefix), 0, -1)
	managedlist = [int(i.decode()) for i in managed]
	managedlist.sort()
	return managedlist

def getIDLEGPUs(rs, prefix):
	idle = rs.lrange("%s_IDLE"%(prefix), 0, -1)
	idlelist = [int(i.decode()) for i in idle]
	idlelist.sort()
	return idlelist

def getActiveGPUs(rs, prefix):
	managedlist = getManagedGPUs(rs, prefix)
	idlelist = getIDLEGPUs(rs, prefix)
	active = [i for i in managedlist if i not in idlelist]
	active.sort()
	return active

def getGPUCount():
	drv.init()
	return drv.Device.count()


if ARG.gpu_count:
	drv.init()
	GPUcount = drv.Device.count()
	print(GPUcount)
	exit()

def insertGPU(rs, prefix, GPUID):
	GPUCOUNT = getGPUCount()
	if GPUID >= GPUCOUNT or GPUID < 0:
		print("GPU ID %d is out of range."%(GPUID))
		exit()
	managed = getManagedGPUs(rs, prefix)
	if GPUID in managed:
		print("GPU %d is already managed."%(GPUID))
		exit()
	else:
		msg = "%s: GPU %d is now managed."%(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), GPUID)
		rs.lpush("%s_EVENTS"%(prefix), msg)
		rs.lpush("%s_MANAGED"%(prefix), GPUID)
		rs.lpush("%s_IDLE"%(prefix), GPUID)
		print(msg)
		exit()

def removeGPU(rs, prefix, GPUID):
	managed = getManagedGPUs(rs, prefix)
	if GPUID not in managed:
		print("GPU %d is not managed."%(GPUID))
		exit()
	else:
		msg = "%s: GPU %d is no longer managed."%(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), GPUID)
		rs.lpush("%s_EVENTS"%(prefix), msg)
		rs.lrem("%s_MANAGED"%(prefix), 0, GPUID)
		rs.lrem("%s_IDLE"%(prefix), 0, GPUID)
		print(msg)
		exit()

if ARG.platform_id:
	cuda_devices = []
	drv.init()
	for i in range(drv.Device.count()):
		device = drv.Device(i)
		cuda_devices.append(device.name())
# List all OpenCL platforms and devices
	platforms = cl.get_platforms()
	for ID, platform in enumerate(platforms):
		print(f"Platform {platform.platform_id}: {platform.name}")
		for device in platform.get_devices():
			print(platform.platform_id)
			exit()
	print("Platform not found")
	exit(-1)

def main():
	parser = argparse.ArgumentParser(description="GPU management through Redis server")
	parser.add_argument("--gpu-count", action="store_true", help="Get number of GPUs in the system as returned by pycuda.driver.Device.count()")
	parser.add_argument("--managed-gpu-count", action="store_true", help="Get number of GPUs managed by the Redis server")
	
	
	parser.add_argument("--redis-server", type=str, default="localhost", help="Redis server address")
	parser.add_argument("--redis-port", type=int, default=6379, help="Redis server port")
	parser.add_argument("--redis-db", type=int, default=0, help="Redis server db")
	
	parser.add_argument("--redis-prefix", type=str, default="GPU", help="Prefix of Redis objects to be used for GPU management")
	manage = parser.add_mutually_exclusive_group()
	manage.add_argument("--redis-manage-all", action="store_true", help="Initialize GPU_IDLE, GPU_MANAGED and GPU_EVENTS objects with all available GPU.")
	manage.add_argument("--redis-manage", type=str, help="Initialize GPU_IDLE, GPU_MANAGED and GPU_EVENTS objects with specified GPUs provided as comma separated list.")
	manage.add_argument("--redis-manage-count", type=int, help="Initialize GPU_IDLE, GPU_MANAGED and GPU_EVENTS objects with specified GPUs provided as comma separated list.")
	parser.add_argument("--force", action="store_true", help="Initialize Redis objects even if they are already initialized.")
	
	parser.add_argument("--pid", type=int, default=None, help="Use this PID to identify PID for --get and --release instead of parent PID.")
	parser.add_argument("--get", action="store_true", help="Get GPU ID that is not in use.")
	parser.add_argument("--release", default=None, type=int, help="Release GPU ID after it is no longer in use")
	
	parser.add_argument("--log", action="store_true", help="Print log.")
	parser.add_argument("--info", action="store_true", help="Print GPUs summary.")
	parser.add_argument("--platform-id", action="store_true", help="Print Platform ID with CUDA GPUs.")
	parser.add_argument("--idle", action="store_true", help="Print idle GPUs.")
	parser.add_argument("--managed", action="store_true", help="Print managed GPUs.")
	parser.add_argument("--active", action="store_true", help="Print active GPUs.")
	parser.add_argument("--redis-delete", action="store_true", help="Delete all Redis objects related to GPU management.")
	parser.add_argument("--redis-purge", action="store_true", help="Remove allocations by non existent processes.")
	
	parser.add_argument("--redis-managed-insert", default=None, type=int, help="Insert GPU ID to managed IDs.")
	parser.add_argument("--redis-managed-delete", default=None, type=int, help="Remove GPU ID from managed IDs.")
	ARG = parser.parse_args()
	
	rs = redis.Redis(host=ARG.redis_server, port=ARG.redis_port, db=ARG.redis_db)
	try:
		rs.ping()
	except redis.ConnectionError:
		print("Can not connect to Redis server, try to start server first.")
		exit(-1)
	
	
	if ARG.managed_gpu_count:
		devCount = len(rs.lrange("%s_MANAGED"%(ARG.redis_prefix), 0, -1))
		print(devCount)
		exit()
	
	if ARG.redis_manage_all or ARG.redis_manage or ARG.redis_manage_count:
		drv.init()
		GPUcount = drv.Device.count()
		managedList = []
		if ARG.redis_manage_all:
			managedList = list(range(GPUcount))
		elif ARG.redis_manage:
			managedList = ARG.redis_manage.split(",")
			managedList = [int(i) for i in managedList]
			managedList = list(set(managedList))
			managedList = [i for i in managedList if i < GPUcount and i >= 0]
		elif ARG.redis_manage_count:
			managedList = list(range(GPUcount))
			managedList = reversed(managedList)#To make just last GPUs managed
			if ARG.redis_manage_count < GPUcount:
				managedList = managedList[:ARG.redis_manage_count]
			else:
				print("Can not manage more GPUs than available but you requested %d out of %d possible!"%(ARG.redis_manage_count, GPUcount))
				exit(-1)
		status = initRedisObjects(rs, ARG.redis_prefix, managedList, GPUcount, ARG.force)
		if status == -1:
			exit()
		else:
			print(len(managedList))
			exit()
	
	if ARG.redis_managed_insert is not None:
		insertGPU(rs, ARG.redis_prefix, ARG.redis_managed_insert)
		exit()
	
	if ARG.redis_managed_delete is not None:
		removeGPU(rs, ARG.redis_prefix, ARG.redis_managed_delete)
		exit()
	
	if ARG.get:
		if ARG.pid is not None:
			PID = ARG.pid
		else:
			PID = os.getppid()
		GPUID = rs.brpop("%s_IDLE"%(ARG.redis_prefix), 0)
		while True:
			if GPUID is not None:
				ID = int(GPUID[1].decode())
				mangedList = getManagedGPUs(rs, ARG.redis_prefix)
				if ID in mangedList:
					LOG = "%s: GPU %s acquired by PID %d."%(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ID, PID)
					rs.lpush("%s_EVENTS"%(ARG.redis_prefix), LOG)
					rs.hset("%s_GPU%d"%(ARG.redis_prefix, int(ID)), "PID", PID)
					rs.hset("%s_GPU%d"%(ARG.redis_prefix, int(ID)), "TIME", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
					print(ID)
					break
			GPUID = rs.brpop("%s_IDLE"%(ARG.redis_prefix), 0)
		exit()
	
	if ARG.release is not None:
		ID = int(ARG.release)
		if ARG.pid is not None:
			PID = ARG.pid
		else:
			PID = os.getppid()
		idlelist = getIDLEGPUs(rs, ARG.redis_prefix)
		managedlist = getManagedGPUs(rs, ARG.redis_prefix)
		if ID in idlelist:
			print("GPU %d is already idle."%(ID))
			exit()
		if ID not in managedlist:
			print("GPU %d is not managed."%(ID))
			exit()
		if ID in managedlist and ID not in idlelist:
			TIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			LOG = "%s: GPU %d released by PID %d."%(TIME, ID, PID)
			rs.lpush("%s_EVENTS"%(ARG.redis_prefix), LOG)
			rs.delete("%s_GPU%d"%(ARG.redis_prefix, ID))
			rs.lpush("%s_IDLE"%(ARG.redis_prefix), ID)
			print("GPU %d released."%(ARG.release))
		exit()
	
	if ARG.log:
		events = rs.lrange("%s_EVENTS"%(ARG.redis_prefix), 0, -1)
		for event in reversed(events):
			print(event.decode())
		exit()
	
	if ARG.idle:
		iddlelist = getIDLEGPUs(rs, ARG.redis_prefix)
		print("There is %d IDDLE GPUs, IDs: %s"%(len(iddlelist), ",".join([str(x) for x in iddlelist])))
		exit()
	
	if ARG.managed:
		managedlist = getManagedGPUs(rs, ARG.redis_prefix)
		print("There is %d MANAGED GPUs, IDs: %s"%(len(managedlist), ",".join([str(x) for x in managedlist])))
		exit()
	
	if ARG.info:
		drv.init()
		GPUcount = drv.Device.count()
		managed = rs.lrange("%s_MANAGED"%(ARG.redis_prefix), 0, -1)
		idle = rs.lrange("%s_IDLE"%(ARG.redis_prefix), 0, -1)
		if len(managed) == 0:
			print("GPU total:%d no managed"%(GPUcount))
		else:
			print("GPU total:%d managed:%d idle:%d"%(GPUcount, len(managed), len(idle)))
		exit()
	
	if ARG.redis_delete:
		rs.delete("%s_IDLE"%(ARG.redis_prefix))
		rs.delete("%s_MANAGED"%(ARG.redis_prefix))
		rs.delete("%s_EVENTS"%(ARG.redis_prefix))
		rs.delete("%s_INIT"%(ARG.redis_prefix))
		managed = rs.lrange("%s_MANAGED"%(ARG.redis_prefix), 0, -1)
		managedlist = [i.decode() for i in managed]
		for i in managedlist:
			rs.delete("%s_GPU%d"%(ARG.redis_prefix, int(i)))
		print("All Redis objects related to GPU management were deleted.")
		exit()
	
	if ARG.redis_purge:
		activeList = getActiveGPUs(rs, ARG.redis_prefix)
		for i in activeList:
			PID = rs.hget("%s_GPU%d"%(ARG.redis_prefix, int(i)), "PID")
			if PID is not None:
				PID = int(PID.decode())
				if not checkPID(PID):
					rs.delete("%s_GPU%d"%(ARG.redis_prefix, int(i)))
					LOG = "%s: GPU %s acquired by process with PID %d, was released by redis purge as the process with given PID does no longer exist."%(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, PID)
					rs.lpush("%s_EVENTS"%(ARG.redis_prefix), LOG)
					rs.lpush("%s_IDLE"%(ARG.redis_prefix), i)
		exit()
	
	if ARG.active:
		active = getActiveGPUs(rs, ARG.redis_prefix)
		if len(active) == 0:
			print("There are no active GPUs.")
		else:
			print("There is %d ACTIVE GPUs, IDs: %s"%(len(active), ", ".join([str(i) for i in active])))
		for i in active:
			PID = rs.hget("%s_GPU%d"%(ARG.redis_prefix, int(i)), "PID")
			TIME = rs.hget("%s_GPU%d"%(ARG.redis_prefix, int(i)), "TIME")
			if TIME is not None and PID is not None:
				PID = int(PID.decode())
				TIME = TIME.decode()
				if checkPID(PID):
					print(colored("GPU %d acquired by running PID %6d at %s"%(i, PID, TIME), "green"))
				else:
					print(colored("GPU %d acquired by not running PID %6d at %s"%(i, PID, TIME), "red"))
			else:
				print("GPU %d acquired by unknown process."%(i))
		exit()

if __name__ == "__main__":
	main()
