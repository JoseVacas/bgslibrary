/*
This file is part of BGSLibrary.

BGSLibrary is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BGSLibrary is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with BGSLibrary.  If not, see <http://www.gnu.org/licenses/>.
*/
/************************************************************************
BlobResult.cpp

FUNCIONALITAT: Implementaci� de la classe CBlob2Result
AUTOR: Inspecta S.L.
MODIFICACIONS (Modificaci�, Autor, Data):

**************************************************************************/

#include <limits.h>
#include <stdio.h>
#include <functional>
#include <algorithm>
#include "BlobResult.h"
#include "BlobExtraction.h"
//#ifdef _DEBUG
//#include <afx.h>			//suport per a CStrings
//#include <afxwin.h>			//suport per a AfxMessageBox
//#endif

/**************************************************************************
Constructors / Destructors
**************************************************************************/


/**
- FUNCI�: CBlob2Result
- FUNCIONALITAT: Constructor estandard.
- PAR�METRES:
- RESULTAT:
- Crea un CBlob2Result sense cap blob
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 20-07-2004.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/**
- FUNCTION: CBlob2Result
- FUNCTIONALITY: Standard constructor
- PARAMETERS:
- RESULT:
- creates an empty set of blobs
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
CBlob2Result::CBlob2Result()
{
  m_blobs = blob_vector();
}

/**
- FUNCI�: CBlob2Result
- FUNCIONALITAT: Constructor a partir d'una imatge. Inicialitza la seq��ncia de blobs 
amb els blobs resultants de l'an�lisi de blobs de la imatge.
- PAR�METRES:
- source: imatge d'on s'extreuran els blobs
- mask: m�scara a aplicar. Nom�s es calcularan els blobs on la m�scara sigui 
diferent de 0. Els blobs que toquin a un pixel 0 de la m�scara seran 
considerats exteriors.
- threshold: llindar que s'aplicar� a la imatge source abans de calcular els blobs
- findmoments: indica si s'han de calcular els moments de cada blob
- RESULTAT:
- objecte CBlob2Result amb els blobs de la imatge source
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 25-05-2005.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/**
- FUNCTION: CBlob2
- FUNCTIONALITY: Constructor from an image. Fills an object with all the blobs in
the image
- PARAMETERS:
- source: image to extract the blobs from
- mask: optional mask to apply. The blobs will be extracted where the mask is
not 0. All the neighbouring blobs where the mask is 0 will be extern blobs
- threshold: threshold level to apply to the image before computing blobs
- findmoments: true to calculate the blob moments (slower)
- RESULT:
- object with all the blobs in the image. It throws an EXCEPCIO_CALCUL_BLOBS
if some error appears in the BlobAnalysis function
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
CBlob2Result::CBlob2Result(IplImage *source, IplImage *mask, int threshold, bool findmoments)
{
  bool success;

  try
  {
    // cridem la funci� amb el marc a true=1=blanc (aix� no unir� els blobs externs)
    success = BlobAnalysis(source,(uchar)threshold,mask,true,findmoments, m_blobs );
  }
  catch(...)
  {
    success = false;
  }

  if( !success ) throw EXCEPCIO_CALCUL_BLOBS;
}

/**
- FUNCI�: CBlob2Result
- FUNCIONALITAT: Constructor de c�pia. Inicialitza la seq��ncia de blobs 
amb els blobs del par�metre.
- PAR�METRES:
- source: objecte que es copiar�
- RESULTAT:
- objecte CBlob2Result amb els blobs de l'objecte source
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 25-05-2005.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/**
- FUNCTION: CBlob2Result
- FUNCTIONALITY: Copy constructor
- PARAMETERS:
- source: object to copy
- RESULT:
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
CBlob2Result::CBlob2Result( const CBlob2Result &source )
{
  m_blobs = blob_vector( source.GetNumBlobs() );

  // creem el nou a partir del passat com a par�metre
  m_blobs = blob_vector( source.GetNumBlobs() );
  // copiem els blobs de l'origen a l'actual
  blob_vector::const_iterator pBlobsSrc = source.m_blobs.begin();
  blob_vector::iterator pBlobsDst = m_blobs.begin();

  while( pBlobsSrc != source.m_blobs.end() )
  {
    // no podem cridar a l'operador = ja que blob_vector �s un 
    // vector de CBlob2*. Per tant, creem un blob nou a partir del
    // blob original
    *pBlobsDst = new CBlob2(**pBlobsSrc);
    pBlobsSrc++;
    pBlobsDst++;
  }
}



/**
- FUNCI�: ~CBlob2Result
- FUNCIONALITAT: Destructor estandard.
- PAR�METRES:
- RESULTAT:
- Allibera la mem�ria reservada de cadascun dels blobs de la classe
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 25-05-2005.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/**
- FUNCTION: ~CBlob2Result
- FUNCTIONALITY: Destructor
- PARAMETERS:
- RESULT:
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
CBlob2Result::~CBlob2Result()
{
  ClearBlobs();
}

/**************************************************************************
Operadors / Operators
**************************************************************************/


/**
- FUNCI�: operador =
- FUNCIONALITAT: Assigna un objecte source a l'actual
- PAR�METRES:
- source: objecte a assignar
- RESULTAT:
- Substitueix els blobs actuals per els de l'objecte source
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 25-05-2005.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/**
- FUNCTION: Assigment operator
- FUNCTIONALITY: 
- PARAMETERS:
- RESULT:
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
CBlob2Result& CBlob2Result::operator=(const CBlob2Result& source)
{
  // si ja s�n el mateix, no cal fer res
  if (this != &source)
  {
    // alliberem el conjunt de blobs antic
    for( int i = 0; i < GetNumBlobs(); i++ )
    {
      delete m_blobs[i];
    }
    m_blobs.clear();
    // creem el nou a partir del passat com a par�metre
    m_blobs = blob_vector( source.GetNumBlobs() );
    // copiem els blobs de l'origen a l'actual
    blob_vector::const_iterator pBlobsSrc = source.m_blobs.begin();
    blob_vector::iterator pBlobsDst = m_blobs.begin();

    while( pBlobsSrc != source.m_blobs.end() )
    {
      // no podem cridar a l'operador = ja que blob_vector �s un 
      // vector de CBlob2*. Per tant, creem un blob nou a partir del
      // blob original
      *pBlobsDst = new CBlob2(**pBlobsSrc);
      pBlobsSrc++;
      pBlobsDst++;
    }
  }
  return *this;
}


/**
- FUNCI�: operador +
- FUNCIONALITAT: Concatena els blobs de dos CBlob2Result
- PAR�METRES:
- source: d'on s'agafaran els blobs afegits a l'actual
- RESULTAT:
- retorna un nou CBlob2Result amb els dos CBlob2Result concatenats
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 25-05-2005.
- NOTA: per la implementaci�, els blobs del par�metre es posen en ordre invers
- MODIFICACI�: Data. Autor. Descripci�.
*/
/**
- FUNCTION: + operator
- FUNCTIONALITY: Joins the blobs in source with the current ones
- PARAMETERS:
- source: object to copy the blobs
- RESULT:
- object with the actual blobs and the source blobs
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
CBlob2Result CBlob2Result::operator+( const CBlob2Result& source )
{	
  //creem el resultat a partir dels blobs actuals
  CBlob2Result resultat( *this );

  // reservem mem�ria per als nous blobs
  resultat.m_blobs.resize( resultat.GetNumBlobs() + source.GetNumBlobs() );

  // declarem els iterador per rec�rrer els blobs d'origen i desti
  blob_vector::const_iterator pBlobsSrc = source.m_blobs.begin();
  blob_vector::iterator pBlobsDst = resultat.m_blobs.end();

  // insertem els blobs de l'origen a l'actual
  while( pBlobsSrc != source.m_blobs.end() )
  {
    pBlobsDst--;
    *pBlobsDst = new CBlob2(**pBlobsSrc);
    pBlobsSrc++;
  }

  return resultat;
}

/**************************************************************************
Operacions / Operations
**************************************************************************/

/**
- FUNCI�: AddBlob
- FUNCIONALITAT: Afegeix un blob al conjunt
- PAR�METRES:
- blob: blob a afegir
- RESULTAT:
- modifica el conjunt de blobs actual
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 2006/03/01
- MODIFICACI�: Data. Autor. Descripci�.
*/
void CBlob2Result::AddBlob( CBlob2 *blob )
{
  if( blob != NULL )
    m_blobs.push_back( new CBlob2( blob ) );
}


#ifdef MATRIXCV_ACTIU

/**
- FUNCI�: GetResult
- FUNCIONALITAT: Calcula el resultat especificat sobre tots els blobs de la classe
- PAR�METRES:
- evaluador: Qualsevol objecte derivat de COperadorBlob
- RESULTAT:
- Retorna un array de double's amb el resultat per cada blob
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 25-05-2005.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/**
- FUNCTION: GetResult
- FUNCTIONALITY: Computes the function evaluador on all the blobs of the class
and returns a vector with the result
- PARAMETERS:
- evaluador: function to apply to each blob (any object derived from the 
COperadorBlob class )
- RESULT:
- vector with all the results in the same order as the blobs
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
double_vector CBlob2Result::GetResult( funcio_calculBlob *evaluador ) const
{
  if( GetNumBlobs() <= 0 )
  {
    return double_vector();
  }

  // definim el resultat
  double_vector result = double_vector( GetNumBlobs() );
  // i iteradors sobre els blobs i el resultat
  double_vector::iterator itResult = result.GetIterator();
  blob_vector::const_iterator itBlobs = m_blobs.begin();

  // avaluem la funci� en tots els blobs
  while( itBlobs != m_blobs.end() )
  {
    *itResult = (*evaluador)(**itBlobs);
    itBlobs++;
    itResult++;
  }
  return result;
}
#endif

/**
- FUNCI�: GetSTLResult
- FUNCIONALITAT: Calcula el resultat especificat sobre tots els blobs de la classe
- PAR�METRES:
- evaluador: Qualsevol objecte derivat de COperadorBlob
- RESULTAT:
- Retorna un array de double's STL amb el resultat per cada blob
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 25-05-2005.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/**
- FUNCTION: GetResult
- FUNCTIONALITY: Computes the function evaluador on all the blobs of the class
and returns a vector with the result
- PARAMETERS:
- evaluador: function to apply to each blob (any object derived from the 
COperadorBlob class )
- RESULT:
- vector with all the results in the same order as the blobs
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
double_stl_vector CBlob2Result::GetSTLResult( funcio_calculBlob *evaluador ) const
{
  if( GetNumBlobs() <= 0 )
  {
    return double_stl_vector();
  }

  // definim el resultat
  double_stl_vector result = double_stl_vector( GetNumBlobs() );
  // i iteradors sobre els blobs i el resultat
  double_stl_vector::iterator itResult = result.begin();
  blob_vector::const_iterator itBlobs = m_blobs.begin();

  // avaluem la funci� en tots els blobs
  while( itBlobs != m_blobs.end() )
  {
    *itResult = (*evaluador)(**itBlobs);
    itBlobs++;
    itResult++;
  }
  return result;
}

/**
- FUNCI�: GetNumber
- FUNCIONALITAT: Calcula el resultat especificat sobre un �nic blob de la classe
- PAR�METRES:
- evaluador: Qualsevol objecte derivat de COperadorBlob
- indexblob: n�mero de blob del que volem calcular el resultat.
- RESULTAT:
- Retorna un double amb el resultat
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 25-05-2005.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/**
- FUNCTION: GetNumber
- FUNCTIONALITY: Computes the function evaluador on a blob of the class
- PARAMETERS:
- indexBlob: index of the blob to compute the function
- evaluador: function to apply to each blob (any object derived from the 
COperadorBlob class )
- RESULT:
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
double CBlob2Result::GetNumber( int indexBlob, funcio_calculBlob *evaluador ) const
{
  if( indexBlob < 0 || indexBlob >= GetNumBlobs() )
    RaiseError( EXCEPTION_BLOB_OUT_OF_BOUNDS );
  return (*evaluador)( *m_blobs[indexBlob] );
}

/////////////////////////// FILTRAT DE BLOBS ////////////////////////////////////

/**
- FUNCI�: Filter
- FUNCIONALITAT: Filtra els blobs de la classe i deixa el resultat amb nom�s 
els blobs que han passat el filtre.
El filtrat es basa en especificar condicions sobre un resultat dels blobs
i seleccionar (o excloure) aquells blobs que no compleixen una determinada
condicio
- PAR�METRES:
- dst: variable per deixar els blobs filtrats
- filterAction:	acci� de filtrat. Incloure els blobs trobats (B_INCLUDE),
o excloure els blobs trobats (B_EXCLUDE)
- evaluador: Funci� per evaluar els blobs (qualsevol objecte derivat de COperadorBlob
- Condition: tipus de condici� que ha de superar la mesura (FilterType) 
sobre cada blob per a ser considerat.
B_EQUAL,B_NOT_EQUAL,B_GREATER,B_LESS,B_GREATER_OR_EQUAL,
B_LESS_OR_EQUAL,B_INSIDE,B_OUTSIDE
- LowLimit:  valor num�ric per a la comparaci� (Condition) de la mesura (FilterType)
- HighLimit: valor num�ric per a la comparaci� (Condition) de la mesura (FilterType)
(nom�s t� sentit per a aquelles condicions que tenen dos valors 
(B_INSIDE, per exemple).
- RESULTAT:
- Deixa els blobs resultants del filtrat a destination
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 25-05-2005.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/**
- FUNCTION: Filter
- FUNCTIONALITY: Get some blobs from the class based on conditions on measures
of the blobs. 
- PARAMETERS:
- dst: where to store the selected blobs
- filterAction:	B_INCLUDE: include the blobs which pass the filter in the result 
B_EXCLUDE: exclude the blobs which pass the filter in the result 
- evaluador: Object to evaluate the blob
- Condition: How to decide if  the result returned by evaluador on each blob
is included or not. It can be:
B_EQUAL,B_NOT_EQUAL,B_GREATER,B_LESS,B_GREATER_OR_EQUAL,
B_LESS_OR_EQUAL,B_INSIDE,B_OUTSIDE
- LowLimit:  numerical value to evaluate the Condition on evaluador(blob)
- HighLimit: numerical value to evaluate the Condition on evaluador(blob).
Only useful for B_INSIDE and B_OUTSIDE
- RESULT:
- It returns on dst the blobs that accomplish (B_INCLUDE) or discards (B_EXCLUDE)
the Condition on the result returned by evaluador on each blob
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
void CBlob2Result::Filter(CBlob2Result &dst, 
                         int filterAction, 
                         funcio_calculBlob *evaluador, 
                         int condition, 
                         double lowLimit, double highLimit /*=0*/)

{
  int i, numBlobs;
  bool resultavaluacio;
  double_stl_vector avaluacioBlobs;
  double_stl_vector::iterator itavaluacioBlobs;

  if( GetNumBlobs() <= 0 ) return;
  if( !evaluador ) return;
  //avaluem els blobs amb la funci� pertinent	
  avaluacioBlobs = GetSTLResult(evaluador);
  itavaluacioBlobs = avaluacioBlobs.begin();
  numBlobs = GetNumBlobs();
  switch(condition)
  {
  case B_EQUAL:
    for(i=0;i<numBlobs;i++, itavaluacioBlobs++)
    {
      resultavaluacio= *itavaluacioBlobs == lowLimit;
      if( ( resultavaluacio && filterAction == B_INCLUDE ) ||
        ( !resultavaluacio && filterAction == B_EXCLUDE ))
      {
        dst.m_blobs.push_back( new CBlob2( GetBlob( i ) ));
      }				
    }
    break;
  case B_NOT_EQUAL:
    for(i=0;i<numBlobs;i++, itavaluacioBlobs++)
    {
      resultavaluacio = *itavaluacioBlobs != lowLimit;
      if( ( resultavaluacio && filterAction == B_INCLUDE ) ||
        ( !resultavaluacio && filterAction == B_EXCLUDE ))
      {
        dst.m_blobs.push_back( new CBlob2( GetBlob( i ) ));
      }
    }
    break;
  case B_GREATER:
    for(i=0;i<numBlobs;i++, itavaluacioBlobs++)
    {
      resultavaluacio= *itavaluacioBlobs > lowLimit;
      if( ( resultavaluacio && filterAction == B_INCLUDE ) ||
        ( !resultavaluacio && filterAction == B_EXCLUDE ))
      {
        dst.m_blobs.push_back( new CBlob2( GetBlob( i ) ));
      }
    }
    break;
  case B_LESS:
    for(i=0;i<numBlobs;i++, itavaluacioBlobs++)
    {
      resultavaluacio= *itavaluacioBlobs < lowLimit;
      if( ( resultavaluacio && filterAction == B_INCLUDE ) ||
        ( !resultavaluacio && filterAction == B_EXCLUDE ))
      {
        dst.m_blobs.push_back( new CBlob2( GetBlob( i ) ));
      }
    }
    break;
  case B_GREATER_OR_EQUAL:
    for(i=0;i<numBlobs;i++, itavaluacioBlobs++)
    {
      resultavaluacio= *itavaluacioBlobs>= lowLimit;
      if( ( resultavaluacio && filterAction == B_INCLUDE ) ||
        ( !resultavaluacio && filterAction == B_EXCLUDE ))
      {
        dst.m_blobs.push_back( new CBlob2( GetBlob( i ) ));
      }
    }
    break;
  case B_LESS_OR_EQUAL:
    for(i=0;i<numBlobs;i++, itavaluacioBlobs++)
    {
      resultavaluacio= *itavaluacioBlobs <= lowLimit;
      if( ( resultavaluacio && filterAction == B_INCLUDE ) ||
        ( !resultavaluacio && filterAction == B_EXCLUDE ))
      {
        dst.m_blobs.push_back( new CBlob2( GetBlob( i ) ));
      }
    }
    break;
  case B_INSIDE:
    for(i=0;i<numBlobs;i++, itavaluacioBlobs++)
    {
      resultavaluacio=( *itavaluacioBlobs >= lowLimit) && ( *itavaluacioBlobs <= highLimit); 
      if( ( resultavaluacio && filterAction == B_INCLUDE ) ||
        ( !resultavaluacio && filterAction == B_EXCLUDE ))
      {
        dst.m_blobs.push_back( new CBlob2( GetBlob( i ) ));
      }
    }
    break;
  case B_OUTSIDE:
    for(i=0;i<numBlobs;i++, itavaluacioBlobs++)
    {
      resultavaluacio=( *itavaluacioBlobs < lowLimit) || ( *itavaluacioBlobs > highLimit); 
      if( ( resultavaluacio && filterAction == B_INCLUDE ) ||
        ( !resultavaluacio && filterAction == B_EXCLUDE ))
      {
        dst.m_blobs.push_back( new CBlob2( GetBlob( i ) ));
      }
    }
    break;
  }


  // en cas de voler filtrar un CBlob2Result i deixar-ho en el mateix CBlob2Result
  // ( operacio inline )
  if( &dst == this ) 
  {
    // esborrem els primers blobs ( que s�n els originals )
    // ja que els tindrem replicats al final si passen el filtre
    blob_vector::iterator itBlobs = m_blobs.begin();
    for( int i = 0; i < numBlobs; i++ )
    {
      delete *itBlobs;
      itBlobs++;
    }
    m_blobs.erase( m_blobs.begin(), itBlobs );
  }
}


/**
- FUNCI�: GetBlob
- FUNCIONALITAT: Retorna un blob si aquest existeix (index != -1)
- PAR�METRES:
- indexblob: index del blob a retornar
- RESULTAT:
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 25-05-2005.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/*
- FUNCTION: GetBlob
- FUNCTIONALITY: Gets the n-th blob (without ordering the blobs)
- PARAMETERS:
- indexblob: index in the blob array
- RESULT:
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
CBlob2 CBlob2Result::GetBlob(int indexblob) const
{	
  if( indexblob < 0 || indexblob >= GetNumBlobs() )
    RaiseError( EXCEPTION_BLOB_OUT_OF_BOUNDS );

  return *m_blobs[indexblob];
}
CBlob2 *CBlob2Result::GetBlob(int indexblob)
{	
  if( indexblob < 0 || indexblob >= GetNumBlobs() )
    RaiseError( EXCEPTION_BLOB_OUT_OF_BOUNDS );

  return m_blobs[indexblob];
}

/**
- FUNCI�: GetNthBlob
- FUNCIONALITAT: Retorna l'en�ssim blob segons un determinat criteri
- PAR�METRES:
- criteri: criteri per ordenar els blobs (objectes derivats de COperadorBlob)
- nBlob: index del blob a retornar
- dst: on es retorna el resultat
- RESULTAT:
- retorna el blob nBlob a dst ordenant els blobs de la classe segons el criteri
en ordre DESCENDENT. Per exemple, per obtenir el blob major:
GetNthBlob( CBlob2GetArea(), 0, blobMajor );
GetNthBlob( CBlob2GetArea(), 1, blobMajor ); (segon blob m�s gran)
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 25-05-2005.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/*
- FUNCTION: GetNthBlob
- FUNCTIONALITY: Gets the n-th blob ordering first the blobs with some criteria
- PARAMETERS:
- criteri: criteria to order the blob array
- nBlob: index of the returned blob in the ordered blob array
- dst: where to store the result
- RESULT:
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
void CBlob2Result::GetNthBlob( funcio_calculBlob *criteri, int nBlob, CBlob2 &dst ) const
{
  // verifiquem que no estem accedint fora el vector de blobs
  if( nBlob < 0 || nBlob >= GetNumBlobs() )
  {
    //RaiseError( EXCEPTION_BLOB_OUT_OF_BOUNDS );
    dst = CBlob2();
    return;
  }

  double_stl_vector avaluacioBlobs, avaluacioBlobsOrdenat;
  double valorEnessim;

  //avaluem els blobs amb la funci� pertinent	
  avaluacioBlobs = GetSTLResult(criteri);

  avaluacioBlobsOrdenat = double_stl_vector( GetNumBlobs() );

  // obtenim els nBlob primers resultats (en ordre descendent)
  std::partial_sort_copy( avaluacioBlobs.begin(), 
    avaluacioBlobs.end(),
    avaluacioBlobsOrdenat.begin(), 
    avaluacioBlobsOrdenat.end(),
    std::greater<double>() );

  valorEnessim = avaluacioBlobsOrdenat[nBlob];

  // busquem el primer blob que t� el valor n-ssim
  double_stl_vector::const_iterator itAvaluacio = avaluacioBlobs.begin();

  bool trobatBlob = false;
  int indexBlob = 0;
  while( itAvaluacio != avaluacioBlobs.end() && !trobatBlob )
  {
    if( *itAvaluacio == valorEnessim )
    {
      trobatBlob = true;
      dst = CBlob2( GetBlob(indexBlob));
    }
    itAvaluacio++;
    indexBlob++;
  }
}

/**
- FUNCI�: ClearBlobs
- FUNCIONALITAT: Elimina tots els blobs de l'objecte
- PAR�METRES:
- RESULTAT: 
- Allibera tota la mem�ria dels blobs
- RESTRICCIONS:
- AUTOR: Ricard Borr�s Navarra
- DATA DE CREACI�: 25-05-2005.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/*
- FUNCTION: ClearBlobs
- FUNCTIONALITY: Clears all the blobs from the object and releases all its memory
- PARAMETERS:
- RESULT:
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
void CBlob2Result::ClearBlobs()
{
  /*for( int i = 0; i < GetNumBlobs(); i++ )
  {
  delete m_blobs[i];
  }*/
  blob_vector::iterator itBlobs = m_blobs.begin();
  while( itBlobs != m_blobs.end() )
  {
    delete *itBlobs;
    itBlobs++;
  }

  m_blobs.clear();
}

/**
- FUNCI�: RaiseError
- FUNCIONALITAT: Funci� per a notificar errors al l'usuari (en debug) i llen�a
les excepcions
- PAR�METRES:
- errorCode: codi d'error
- RESULTAT: 
- Ensenya un missatge a l'usuari (en debug) i llen�a una excepci�
- RESTRICCIONS:
- AUTOR: Ricard Borr�s Navarra
- DATA DE CREACI�: 25-05-2005.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/*
- FUNCTION: RaiseError
- FUNCTIONALITY: Error handling function
- PARAMETERS:
- errorCode: reason of the error
- RESULT:
- in _DEBUG version, shows a message box with the error. In release is silent.
In both cases throws an exception with the error.
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
void CBlob2Result::RaiseError(const int errorCode) const
{
  // estem en mode debug?
//#ifdef _DEBUG
//  CString msg, format = "Error en CBlob2Result: %s";
//
//  switch (errorCode)
//  {
//  case EXCEPTION_BLOB_OUT_OF_BOUNDS:
//    msg.Format(format, "Intentant accedir a un blob no existent");
//    break;
//  default:
//    msg.Format(format, "Codi d'error desconegut");
//    break;
//  }
//
//  AfxMessageBox(msg);
//
//#endif
  throw errorCode;
}



/**************************************************************************
Auxiliars / Auxiliary functions
**************************************************************************/


/**
- FUNCI�: PrintBlobs
- FUNCIONALITAT: Escriu els par�metres (�rea, per�metre, exterior, mitjana) 
de tots els blobs a un fitxer.
- PAR�METRES:
- nom_fitxer: path complet del fitxer amb el resultat
- RESULTAT:
- RESTRICCIONS:
- AUTOR: Ricard Borr�s
- DATA DE CREACI�: 25-05-2005.
- MODIFICACI�: Data. Autor. Descripci�.
*/
/*
- FUNCTION: PrintBlobs
- FUNCTIONALITY: Prints some blob features in an ASCII file
- PARAMETERS:
- nom_fitxer: full path + filename to generate
- RESULT:
- RESTRICTIONS:
- AUTHOR: Ricard Borr�s
- CREATION DATE: 25-05-2005.
- MODIFICATION: Date. Author. Description.
*/
void CBlob2Result::PrintBlobs( char *nom_fitxer ) const
{
  double_stl_vector area, /*perimetre,*/ exterior, mitjana, compacitat, longitud, 
    externPerimeter, perimetreConvex, perimetre;
  int i;
  FILE *fitxer_sortida;

  area      = GetSTLResult( CBlob2GetArea());
  perimetre = GetSTLResult( CBlob2GetPerimeter());
  exterior  = GetSTLResult( CBlob2GetExterior());
  mitjana   = GetSTLResult( CBlob2GetMean());
  compacitat = GetSTLResult(CBlob2GetCompactness());
  longitud  = GetSTLResult( CBlob2GetLength());
  externPerimeter = GetSTLResult( CBlob2GetExternPerimeter());
  perimetreConvex = GetSTLResult( CBlob2GetHullPerimeter());

  fitxer_sortida = fopen( nom_fitxer, "w" );

  for(i=0; i<GetNumBlobs(); i++)
  {
    fprintf( fitxer_sortida, "blob %d ->\t a=%7.0f\t p=%8.2f (%8.2f extern)\t pconvex=%8.2f\t ext=%.0f\t m=%7.2f\t c=%3.2f\t l=%8.2f\n",
      i, area[i], perimetre[i], externPerimeter[i], perimetreConvex[i], exterior[i], mitjana[i], compacitat[i], longitud[i] );
  }
  fclose( fitxer_sortida );

}
