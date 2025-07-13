#include <stdio.h>
#include <string.h>
#include <errno.h>
 
#include <wiringPi.h>
#include <wiringSerial.h>
 
#include <pthread.h>
#include <stdlib.h>
 
void *read_serial(void *arg)
{
    char *sendbuf;
    sendbuf = (char *)malloc(32*sizeof(char));
    char *p = sendbuf;
 
    while(1){
        memset(sendbuf,'\0',32*sizeof(char));
        fgets(sendbuf,sizeof(sendbuf),stdin);
        //scanf("%s",sendbuf);
        while(*sendbuf != '\0'){
            serialPutchar (*((int *)arg), *sendbuf) ; //串口打印数据的函数 serialPutchar()
            sendbuf++;
        }
        sendbuf = p;
    }
 
    pthread_exit(NULL);
 
}
 
void *write_serial(void *arg)
{
	while(1){
		while(serialDataAvail (*((int *)arg))){ //当串口有数据的时候进入while
			printf ("%c", serialGetchar (*((int *)arg))) ; //串口接收数据的函数serialGetchar()
			fflush (stdout) ;
		}
	}
 
	pthread_exit(NULL);
}
 
 
 
int main ()
{
	int fd ;
	int ret;
	pthread_t read_thread;
	pthread_t write_thread;
 
	if ((fd = serialOpen ("/dev/ttyS0", 115200)) < 0) //打开驱动文件，配置波特率
	{
		fprintf (stderr, "Unable to open serial device: %s\n", strerror (errno)) ;
		return 1 ;
	}
 
	if (wiringPiSetup () == -1)
	{
		fprintf (stdout, "Unable to start wiringPi: %s\n", strerror (errno)) ;
		return 1 ;
	}
 
	ret = pthread_create(&read_thread,NULL,read_serial,(void *)&fd);
	if(ret != 0){
		printf("read_serial create error\n");
		return 1;
	}
	ret = pthread_create(&write_thread,NULL,write_serial,(void *)&fd);
	if(ret != 0){
		printf("write_serial create error\n");
		return 1;
	}
 
	pthread_join(read_thread,NULL);
	pthread_join(write_thread,NULL);
 
	return 0 ;
}