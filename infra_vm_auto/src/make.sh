#!/usr/bin/env bash

# Parameters of the VM
VM_NAME="regnilo-debian12"
OS_VARIANT="debian12"
RAM="2048"
DISK_SIZE="10"
DI_PATH=http://ftp.debian.org/debian/dists/bookworm/main/installer-amd64/
VM_IMAGE_PATH="/var/lib/libvirt/images/$VM_NAME.qcow2"

# Check if the VM already exists
if virsh list --all | grep -q "$VM_NAME"; then
  virsh destroy $VM_NAME
  virsh undefine $VM_NAME
  if [ -f "$VM_IMAGE_PATH" ]; then
    rm -f "$VM_IMAGE_PATH"
  fi
fi

# Create the VM
virt-install \
  --name $VM_NAME \
  --ram $RAM \
  --disk path=$VM_IMAGE_PATH,size=$DISK_SIZE,format=qcow2 \
  --vcpus 2 \
  --os-variant $OS_VARIANT \
  --graphics spice \
  --location $DI_PATH \
  --initrd-inject=preseed.cfg \
  --initrd-inject=postinst.sh \
  --noautoconsole \
  --wait -1

# Automatic SSH login
./auto_login.sh
