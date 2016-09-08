/* -*- c++ -*- */
/* 
 * Copyright 2016 <+YOU OR YOUR COMPANY+>.
 * 
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "subcarriers_impl.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <iostream>
namespace gr {
  namespace final {

    subcarriers::sptr
    subcarriers::make(int u_sub, int howlong, char* directory)
    {
      return gnuradio::get_initial_sptr
        (new subcarriers_impl(u_sub, howlong, directory));
    }

    /*
     * The private constructor
     */
    subcarriers_impl::subcarriers_impl(int u_sub, int howlong, char* directory)
      : gr::sync_decimator("subcarriers",
             gr::io_signature::make(1, 1, u_sub*sizeof(float)),
              gr::io_signature::make(0, 0, 0), howlong),num(u_sub),len(howlong)
    {
    	fp=fopen(directory,"w+");
    }

    /*
     * Our virtual destructor.
     */
    subcarriers_impl::~subcarriers_impl()
    {
    }

    int
    subcarriers_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
    {
    const float *in = (const float *) input_items[0];
      //float *out = (float *) output_items[0];
      int i=0;
      int j=0;
      int k=0;
    //  int rank_big=0;
    //  int rank_small=0;
      float temp=0;
      char sub[4000];
      char buff[50];
	  struct timeval ts;
	//unsigned int input_data_size = input_signature()->sizeof_stream_item (0);
     // unsigned int output_data_size = output_signature()->sizeof_stream_item (0);
      //std::cout<<"input: "<<input_data_size<<" output: "<<output_data_size<<" noutput_items "<<noutput_items<<std::endl;
     // std::cout<<"inputsize*num item is "<<in[input_data_size*num]
      for(i=0;i<noutput_items;i++){
      //	rank_big=(int)(i/num);
      //	rank_small=i%num;
      	gettimeofday(&ts,NULL);	
      	for(j=0;j<num;j++){
      		//out[i*num+j]=in[i*len*num+j];
      		for(k=0;k<len;k++){
      			temp+=in[i*num*len+j+k*num];
      		}
      		//out[i*num+j]=temp/len;
      		sprintf(buff,"sub%d,%f,%ld.%ld\n",j,temp/len,ts.tv_sec,ts.tv_usec);
      		temp=0;
      		strcat(sub,buff);
      		//std::cout<<buff<<std::endl;
      		for(k=0;k<50;k++){
      			buff[k]='\0';
      		}
      	}
      	fprintf(fp,sub);
      	for(j=0;j<4000;j++)
      	{
      		sub[i]='\0';

      	}
      	
      }
      return noutput_items;
    }

  } /* namespace final */
} /* namespace gr */

