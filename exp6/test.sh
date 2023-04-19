#!/bin/bash
for file in $(ls | grep "^s\|^g" | grep -v ".cu")
do
  	srun --exclusive $file
done

