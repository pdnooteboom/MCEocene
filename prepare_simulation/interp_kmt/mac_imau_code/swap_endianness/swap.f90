program addrecordseparator
  integer*8, dimension(:),allocatable::a
  integer*8::sz
  integer::n,flag
  CHARACTER(len=128) :: arg
  integer:: values(13)

  call get_command_argument(2,arg)
  read(arg,*)sz
  print*,'File size:',sz
  sz=sz/8
  call get_command_argument(1,arg)
  open(file=arg,unit=11,form='unformatted',access='stream')
  open(file=trim(arg)//'.out',form='unformatted',unit=12,access='stream',convert="swap")
  allocate(a(sz))
    read(11)a
    write(12)a
  close(11)
  close(12)
end

