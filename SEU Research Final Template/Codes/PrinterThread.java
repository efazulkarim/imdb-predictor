/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package syncthread;

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author student
 */
public class PrinterThread implements Runnable{

    public static int a = 0;
    synchronized public void print(int n) 
    {
        try {
            for(int i=0;i<n;i++){
                a ++ ;
                System.out.println("A "+a+": printing "+i+ " in thread "+Thread.currentThread().getName());
                Thread.sleep(100);
            }
        } catch (InterruptedException ex) {
            Logger.getLogger(PrinterThread.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
    @Override
    public void run() {
        print(500);
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void main(String args[])
    {
    
    PrinterThread pthread1 = new PrinterThread();
   
    
    Thread t1 = new Thread(pthread1,"Pthread1");
    Thread t2 = new Thread(pthread1,"Pthread2");
    
    t1.start();
    t2.start();
    
     
    }
}
