# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

from .._libtoast import Logger
from .._libtoast import accel_assign_device as omp_accel_assign_device
from .._libtoast import accel_create as omp_accel_create
from .._libtoast import accel_delete as omp_accel_delete
from .._libtoast import accel_enabled as omp_accel_enabled
from .._libtoast import accel_get_device as omp_accel_get_device
from .._libtoast import accel_present as omp_accel_present
from .._libtoast import accel_update_device as omp_accel_update_device
from .._libtoast import accel_update_host as omp_accel_update_host

enable_vals = ["1", "yes", "true"]
disable_vals = ["0", "no", "false"]

use_accel_omp = False
if "TOAST_GPU_OPENMP" in os.environ and os.environ["TOAST_GPU_OPENMP"] in enable_vals:
    if omp_accel_enabled():
        use_accel_omp = True
    else:
        log = Logger.get()
        msg = "TOAST_GPU_OPENMP enabled at runtime, but package was not built "
        msg += "with OpenMP target offload support."
        log.error(msg)
        raise RuntimeError(msg)

use_accel_jax = False
if ("TOAST_GPU_JAX" in os.environ) and (os.environ["TOAST_GPU_JAX"] in enable_vals):
    try:
        import jax

        from ..jax.device import jax_accel_assign_device, jax_accel_get_device
        from ..jax.intervals import INTERVALS_JAX
        from ..jax.mutableArray import MutableJaxArray

        use_accel_jax = True
    except ImportError:
        # There could be many possible exceptions...
        log = Logger.get()
        msg = "TOAST_GPU_JAX enabled at runtime, but jax is not "
        msg += "importable."
        log.error(msg)
        raise RuntimeError(msg)

if use_accel_omp and use_accel_jax:
    log = Logger.get()
    msg = "OpenMP target offload and JAX cannot both be enabled at runtime."
    log.error(msg)
    raise RuntimeError(msg)

use_hybrid_pipelines = True
if ("TOAST_GPU_HYBRID_PIPELINES" in os.environ) and (
    os.environ["TOAST_GPU_HYBRID_PIPELINES"] in disable_vals
):
    use_hybrid_pipelines = False

# Wrapper functions that work with either numpy arrays mapped to omp device memory
# or jax arrays.


def accel_enabled():
    """Returns True if any accelerator support is enabled."""
    return use_accel_jax or use_accel_omp


def accel_get_device():
    """Return the device ID assigned to this process."""
    if use_accel_omp:
        return omp_accel_get_device()
    elif use_accel_jax:
        return jax_accel_get_device()
    else:
        log = Logger.get()
        log.warning("Accelerator support not enabled, returning device -1")
        return -1


def accel_assign_device(node_procs, node_rank, disabled):
    """
    Assign processes to target devices.

    NOTE:
    One can pick devices visible to processes using Slurm and teh following commands
    `--gpus-per-task=1 --gpu-bind=single:1`

    Args:
        node_procs (int): number of processes per node
        node_rank (int): rank of the current process, within the node
        disabled (bool): gpu computing is disabled

    Returns:
        None: the device is stored in a backend specific global variable
    """
    # FIXME some functions (such as poiting_detector) require the omp device to have been assigned
    # so it should be called even when using JAX or running on CPU
    omp_accel_assign_device(node_procs, node_rank, disabled)
    if use_accel_jax:
        jax_accel_assign_device(node_procs, node_rank, disabled)


def accel_data_present(data):
    """Check if data is present on the device.

    For OpenMP target offload, this checks if the input data has an entry in the
    global map of host to device pointers.
    For jax, this tests if the input array is a jax array.

    Args:
        data (array):  The data to test.

    Returns:
        (bool):  True if the data is present on the device.

    """
    log = Logger.get()
    log.verbose("accel_data_present")
    if data is None:
        return False
    elif use_accel_omp:
        return omp_accel_present(data)
    elif use_accel_jax:
        return (
            isinstance(data, MutableJaxArray)
            or isinstance(data, jax.numpy.ndarray)
            or isinstance(data, INTERVALS_JAX)
        )
    else:
        log.warning("Accelerator support not enabled, data not present")
        return False


def accel_data_create(data):
    """Create device buffers.

    Using the input data array, create a corresponding device array.  For OpenMP
    target offload, this allocates device memory and adds it to the global map
    of host to device pointers. For jax it just wraps the numpy array

    Args:
        data (array):  The host array.

    Returns:
        (object):  Either the original input (for OpenMP) or a new jax array.

    """
    if use_accel_omp:
        omp_accel_create(data)
        return data
    elif use_accel_jax:
        return MutableJaxArray(data)
    else:
        log = Logger.get()
        log.warning("Accelerator support not enabled, cannot create")


def accel_data_update_device(data):
    """Update device buffers.

    Ensure that the input data is updated on the device.  For OpenMP target offload,
    this will do a host to device copy and return the input host object.  For jax,
    this will take the input (either a numpy or jax array) and return a jax array.

    Args:
        data (array):  The host array.

    Returns:
        (object):  Either the original input (for OpenMP) or a new jax array.

    """
    if use_accel_omp:
        omp_accel_update_device(data)
        return data
    elif use_accel_jax:
        return MutableJaxArray(data)
    else:
        log = Logger.get()
        log.warning("Accelerator support not enabled, not updating device")
        return None


def accel_data_update_host(data):
    """Update host buffers.

    Ensure that the input data is updated on the host.  For OpenMP target offload,
    this will do a device to host copy and return the input (updated) host object.
    For jax, this will take the input (either a numpy or jax array) and return a
    numpy array.

    Args:
        data (array):  The host array.

    Returns:
        (object):  Either the updated input (for OpenMP) or a numpy array.

    """
    if use_accel_omp:
        omp_accel_update_host(data)
        return data
    elif use_accel_jax:
        return data.to_host()
    else:
        log = Logger.get()
        log.warning("Accelerator support not enabled, not updating host")


def accel_data_delete(data):
    """Delete device copy of the data.

    For OpenMP target offload, this deletes the device allocated memory and removes
    the host entry from the global memory map.

    For jax, this returns a host array (if needed).

    Args:
        data (array):  The host array.

    Returns:
        None

    """
    if use_accel_omp:
        omp_accel_delete(data)
    elif use_accel_jax:
        # if needed, make sure that data is on host
        if accel_data_present(data):
            data = data.host_data
    else:
        log = Logger.get()
        log.warning("Accelerator support not enabled, cannot delete device data")
    return data


class AcceleratorObject(object):
    """Base class for objects that support offload to an accelerator.

    This provides the API for classes that can move their data to and from one
    of the supported accelerator devices.  The public methods provide a central
    place for docstrings and use the internal methods.  These provide a way to
    add boilerplate and checks in a single place in the code.  The internal
    methods should be overloaded by descendent classes.

    Args:
        None

    """

    def __init__(self):
        # Data always starts off on host
        self._accel_used = False

    def _accel_exists(self):
        return False

    def accel_exists(self):
        """Check if a data copy exists on the accelerator.

        Returns:
            (bool):  True if the data is present.

        """
        if not accel_enabled():
            return False
        return self._accel_exists()

    def accel_in_use(self):
        """Check if the device data copy is the one currently in use.

        Returns:
            (bool):  True if the accelerator device copy is being used.

        """
        return self._accel_used

    def accel_used(self, state):
        """Set the in-use state of the device data copy.

        Setting the state to `True` is only possible if the data exists
        on the device.

        Args:
            state (bool):  True if the device copy is in use, else False.

        Returns:
            None

        """
        if state and not self.accel_exists():
            log = Logger.get()
            msg = f"Data is not present on device, cannot set state to in-use"
            log.error(msg)
            raise RuntimeError(msg)
        self._accel_used = state

    def _accel_create(self):
        msg = f"The _accel_create function was not defined for this class."
        raise RuntimeError(msg)

    def accel_create(self):
        """Create a (potentially uninitialized) copy of the data on the accelerator.

        Returns:
            None
        """
        if not accel_enabled():
            return
        if self.accel_exists():
            log = Logger.get()
            msg = f"Data already exists on device, cannot create"
            log.error(msg)
            raise RuntimeError(msg)
        self._accel_create()

    def _accel_update_device(self):
        msg = f"The _accel_update_device function was not defined for this class."
        raise RuntimeError(msg)

    def accel_update_device(self):
        """Copy the data to the accelerator.

        Returns:
            None
        """
        if not accel_enabled():
            return
        if (not self.accel_exists()) and (not use_accel_jax):
            # There is no data on device
            # NOTE: this does no apply to JAX as JAX will allocate on the fly
            log = Logger.get()
            msg = f"Data does not exist on device, cannot update"
            log.error(msg)
            raise RuntimeError(msg)
        if self.accel_in_use():
            # The active copy is already on the device
            log = Logger.get()
            msg = f"Active data is already on device, cannot update"
            log.error(msg)
            raise RuntimeError(msg)
        self._accel_update_device()
        self.accel_used(True)

    def _accel_update_host(self):
        msg = f"The _accel_update_host function was not defined for this class."
        raise RuntimeError(msg)

    def accel_update_host(self):
        """Copy the data to the host from the accelerator.

        Returns:
            None
        """
        if not accel_enabled():
            return
        if not self.accel_exists():
            log = Logger.get()
            msg = f"Data does not exist on device, cannot update host"
            log.error(msg)
            raise RuntimeError(msg)
        if not self.accel_in_use():
            # The active copy is already on the host
            log = Logger.get()
            msg = f"Active data is already on host, cannot update"
            log.error(msg)
            raise RuntimeError(msg)
        self._accel_update_host()
        self.accel_used(False)

    def _accel_delete(self):
        msg = f"The _accel_delete function was not defined for this class."
        raise RuntimeError(msg)

    def accel_delete(self):
        """Delete the data from the accelerator.

        Returns:
            None

        """
        if not accel_enabled():
            return
        if (not self.accel_exists()) and (not use_accel_jax):
            # NOTE: this check does not apply to JAX as the data will not be on device after an update_host
            log = Logger.get()
            msg = f"Data does not exist on device, cannot delete"
            log.error(msg)
            raise RuntimeError(msg)
        self._accel_delete()
        self._accel_used = False