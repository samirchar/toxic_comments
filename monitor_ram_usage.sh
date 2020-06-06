while true;
do free -m | grep Mem >> ram_usage.log; sleep 1;
done
