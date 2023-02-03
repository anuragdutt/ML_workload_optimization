# -*- coding: UTF-8 -*-
# This file is part of the jetson_stats package (https://github.com/rbonghi/jetson_stats or http://rnext.it).
# Copyright (c) 2019 Raffaello Bonghi.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Logging
import logging
import sys
import os
# Launch command
import subprocess as sp
# Threading
from threading import Thread, Event
# Tegrastats parser
from tegra_parse import VALS, MTS, RAM, SWAP, IRAM, CPUS, TEMPS, WATTS
# from .common import locate_commands
# Create logger for tegrastats
logger = logging.getLogger(__name__)


class Tegrastats:


    """
        - Subprocess read:
        https://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python/4896288#4896288
        - Property
        https://www.programiz.com/python-programming/property
    """
    def locate_commands(self,name, commands):
        for cmd in commands:
            if os.path.exists(cmd):
                logger.info("{name} loaded on {cmd}".format(name=name, cmd=cmd))
                return cmd
        raise Exception("{name} is not available on this board".format(name=name))

    def __init__(self, callback, tegrastats_path):
        self._running = Event()
        # Error message from thread
        self._error = None
        # Start process tegrastats
        self.path = self.locate_commands("tegrastats", tegrastats_path)
        # Define Tegrastats process
        self._thread = None
        # Initialize callback
        self.callback = callback

    def _decode(self, text):
        # Find and parse all single values
        stats = VALS(text)
        # Parse if exist MTS
        mts = MTS(text)
        if mts:
            stats['MTS'] = mts
        # Parse RAM
        stats['RAM'] = RAM(text)
        # If exists parse SWAP
        swap = SWAP(text)
        if swap:
            stats['SWAP'] = swap
        # If exists parse IRAM
        iram = IRAM(text)
        if iram:
            stats['IRAM'] = iram
        # Parse CPU status
        stats['CPU'] = CPUS(text)
        # Parse temperatures
        stats['TEMP'] = TEMPS(text)
        # Parse Watts
        stats['WATT'] = WATTS(text)
        return stats



    def _read_tegrastats(self, interval, running):
        pts = sp.Popen([self.path, '--interval', str(interval)], stdout=sp.PIPE)
        try:
            # Reading loop
            while running.is_set():
                if pts.poll() is not None:
                    continue
                out = pts.stdout
                if out is not None:
                    # Read line process output
                    line = out.readline()
                    # Decode line in UTF-8
                    tegrastats_data = line.decode("utf-8")
                    # Decode and store
                    stats = self._decode(tegrastats_data)
                    # Launch callback
                    self.callback(stats)
        except AttributeError:
            pass
        except IOError:
            pass
        except Exception:
            # Write error message
            self._error = sys.exc_info()
        finally:
            # Kill process
            try:
                pts.kill()
            except OSError:
                pass

    def open(self, interval=0.2):
        if self._thread is not None:
            print("Issue")
            return False
        # Set timeout
        interval = int(interval * 1000)
        # Check if thread or process exist
        self._running.set()
        # Start thread Service client
        self._thread = Thread(target=self._read_tegrastats, args=(interval, self._running, ))
        self._thread.start()
        return True

    def close(self, timeout=None):
        # Catch exception if exist
        print("IN close")
        if self._error:
            # Extract exception and raise
            ex_type, ex_value, tb_str = self._error
            ex_value.__traceback__ = tb_str
            raise ex_value
        # Check if thread and process are already empty
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        return True
# EOF